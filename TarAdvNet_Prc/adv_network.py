import pandas as pd
import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl
import torch.nn as nn
from sub_models import *
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
import torch.autograd as autograd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AdversarialNetwork(pl.LightningModule):
    def __init__(self, sample_size, model, n_critic = 5, target_direction = 1, beta1 = 0.0, beta2=0.9, num_days = 400,
                 alpha = 1, init_beta = 1e-05, epoch_betas = [30, 60, 90, 120, 150, 180, 210, 240], beta_scale = 5, c = 5, d = 2, lmda = 1,
                 scale_max = None, scale_min = None, plot_paths = '.', black_box = False):
        super().__init__()

        # General Params
        self.plot_paths = plot_paths
        self.epoch_index = 0
        self.black_box = black_box

        # Scaling Hyperparams
        self.scale_max = scale_max
        self.scale_min = scale_min

        # Model Hyperparaams
        self.sample_size = sample_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.lmda = lmda

        # Loss function hyperparams
        self.alpha = alpha
        self.beta = init_beta
        self.epoch_betas = epoch_betas
        self.beta_scale = beta_scale

        # Adversarial Loss Hyperparams
        if target_direction not in [-1, 1]:
            raise ValueError("Direction should be in the following list: [-1, 1]")
        self.target_direction = target_direction
        self.c = c
        self.d = d
        self.num_days = num_days
        self.forecast_length = model._hparams.prediction_length
        self.lookback_length = model._hparams.context_length

        # Models
        self.generator = Generator(output_dim=1, sample_size=sample_size)
        self.discriminator = Discriminator(input_dim=2, hidden_dim=64, output_dim=1)

        # Logging
        self.step_adv_loss = []
        self.step_f_loss = []
        self.batch_sizes = []
        self.final_losses = []
        self.adversarial_losses = []
        self.discriminator_losses_real = []
        self.discriminator_losses_fake = []
        self.generator_losses = []
        self.batch_sizes = []
        self.w_dist = []
        self.example_outputs = []
        self.gradient_penalties = []
        self.d_loss_real = []
        self.d_loss_fake = []
        self.g_loss = []
        self.final_w_dists = []
        self.g_pens = []


        self.automatic_optimization = False

        #self.save_hyperparameters()

        # Set model to eval mode (no gradient updates)
        object.__setattr__(self, 'model', model)
        self.model.eval()
        # If blackbox, then set grad to false for all nhits params
        if self.black_box:
            for param in self.model.parameters():
                param.requires_grad = False


    '''
    Compute the ema based on the pandas formula in their documentation (when adjust=False)
    '''
    def exponential_moving_average(self, adjprc: torch.Tensor, span: int):

        alpha = 2.0 / (span + 1.0)

        ema = torch.zeros_like(adjprc) 
        ema[0] = adjprc[0]

        # formula taken from the pandas definition
        for i in range(1, len(adjprc)):
            ema[i] = (1 - alpha) * adjprc[i-1] + alpha * adjprc[i]

        return ema
    
    def feature_generation(self, adjprc: torch.Tensor):

        adjprc = adjprc.float()
        weight5 = torch.ones((1,1,5))/5
        weight10 = torch.ones((1,1,10))/10
        weight20 = torch.ones((1,1,20))/20
        rolling_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(1), weight5, stride=1)

        rolling_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(1), weight10, stride=1)

        rolling_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(1), weight20, stride=1)

        # stdevs: Variance = E[x^2] - E[x]^2
        # so E[x] = rolling_meanX => E[x]^2 = rolling_meanX ^ 2
        rollingExp_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(1) ** 2, weight5, stride=1)
        rollingExp_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(1) ** 2, weight10, stride=1)
        rollingExp_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(1) ** 2, weight20, stride=1)

        rolling_stdev5 = torch.sqrt(torch.clip(rollingExp_mean5 - rolling_mean5**2 + 1e-06, min=0))
        rolling_stdev10 = torch.sqrt(torch.clip(rollingExp_mean10 - rolling_mean10**2 + 1e-06, min=0))
        rolling_stdev20 = torch.sqrt(torch.clip(rollingExp_mean20 - rolling_mean20**2 + 1e-06, min=0))
        # but we have padding and the model fills the first x values with something (either adjprc or mean rolling_stdev) so we have to fix that
        rolling_mean5 = rolling_mean5.squeeze(1)
        rolling_mean5 = torch.cat([adjprc[:, :4], rolling_mean5], dim=1)
        rolling_stdev5 = rolling_stdev5.squeeze(1)
        rolling_stdev5 = torch.cat([torch.mean(rolling_stdev20.squeeze(1), dim=1).unsqueeze(-1).repeat(1, 4), rolling_stdev5], dim=1)

        rolling_mean10 = rolling_mean10.squeeze(1)
        rolling_mean10 = torch.cat([adjprc[:, :9], rolling_mean10], dim=1)
        rolling_stdev10 = rolling_stdev10.squeeze(1)
        rolling_stdev10 = torch.cat([torch.mean(rolling_stdev20.squeeze(1), dim=1).unsqueeze(-1).repeat(1, 9), rolling_stdev10], dim=1)

        rolling_mean20 = rolling_mean20.squeeze(1)
        rolling_mean20 = torch.cat([adjprc[:, :19], rolling_mean20], dim=1)
        rolling_stdev20 = rolling_stdev20.squeeze(1)
        rolling_stdev20 = torch.cat([torch.mean(rolling_stdev20.squeeze(1), dim=1).unsqueeze(-1).repeat(1, 19), rolling_stdev20], dim=1)

        # log returns 
        log_returns = torch.log(adjprc[:, 1:]/adjprc[:, :-1]) 
        log_returns = torch.cat([torch.zeros_like(adjprc)[:, :1], log_returns], dim=1) 

        # ROC (percetn change of 5)
        roc5 = (adjprc[:, 5:] - adjprc[:, :-5]) / adjprc[:, :-5]
        roc5 = torch.cat([torch.zeros_like(adjprc)[:, :5], roc5], dim=1)

        # exponential moving averages
        ema_5 = self.exponential_moving_average(adjprc=adjprc, span=5)
        ema_10 = self.exponential_moving_average(adjprc=adjprc, span=10)
        ema_20 = self.exponential_moving_average(adjprc=adjprc, span=20)

        features = torch.stack([rolling_mean5, rolling_mean10, rolling_mean20, 
                                rolling_stdev5, rolling_stdev10, rolling_stdev20, 
                                log_returns, roc5, ema_5, ema_10, ema_20], dim=-1)


        # Because we are doing 1 recording at a time, the standard scalar for teh features (not target) are just the mean and stdev of the recording
        # Similarly, the quantiles are the same for the target for the robust scalar for adjprc

        avg = torch.mean(features, dim=1)
        stdev = torch.std(features, dim=1)

        features = (features - avg.unsqueeze(1)) / stdev.unsqueeze(1)

        # median = torch.median(adjprc, dim=0).values
        # q75 = torch.quantile(adjprc, q=0.75, dim=0)
        # q25 = torch.quantile(adjprc, q=0.25, dim=0)
        # #adjprc = (adjprc - median) / (q75 - q25)

        # scale_ = q75 - q25
        # center_ = median

        center_ = torch.median(adjprc, dim=1).values
        scale_ = (torch.quantile(adjprc, q=0.75, dim=1) - torch.quantile(adjprc, q=0.25, dim=1)) / 2

        features = torch.concat([adjprc.unsqueeze(-1), features], dim=-1)

        return features, scale_, center_
    
    def call_model(self, adjprc: torch.Tensor, days):
        x_prime = adjprc

        features, scale_, center_ = self.feature_generation(x_prime)

        features[:, :, 0] = (features[:, :, 0] - center_.unsqueeze(-1)) / scale_.unsqueeze(-1)

        features = features.type(torch.float32)

        x_prime = x_prime.type(torch.float32)


        # craft payload
        # payload: (dictionary)
        # 1. encoder_cat -> long (batch_size x n_encoder_time_steps x n_features) -> long tensor of encoded categoricals for encoder
        # 2. encoder_cont -> float tensor of scaled continuous variables for encoder
        # 3. encoder_lengths -> long tensor with lengths of the encoder time series. No entry will be greater than n_encoder_time_steps
        # 4. encoder_target -> if list, each entry for a different target. float tensor with unscaled continous target or encoded categorical target, list of tensors for multiple targets
        # 5. decoder_cat -> long tensor of encoded categoricals for decoder
        # 6. decoder_cont -> float tensor of scaled continuous variables for decoder
        # 7. decoder_lengths -> long tensor with lengths of the decoder time series. No entry will be greater than n_decoder_time_steps
        # 8. decoder_target -> if list, with each entry for a different target. float tensor with unscaled continous target or encoded categorical target for decoder - this corresponds to first entry of y, list of tensors for multiple targets
        # 9. target_scale -> if list, with each entry for a different target. parameters used to normalize the target. Typically these are mean and standard deviation. Is list of tensors for multiple targets.

        # for i in range(0, features.shape[1] - 300 - 49): # need to make room for prediction values
        #     encoder_batches.append(features[:, i:i+300, :])
        #     encoder_categorical_batches.append(days[:, i:i+300])
        #     encoder_targets.append(x_prime[:, i:i+300])

        #     decoder_batches.append(features[:, i+300:i+300+50, :])
        #     decoder_categorical_batches.append(days[:, i+300 : i+300+50])
        #     decoder_targets.append(x_prime[:, i+300:i+350])
        #     time_idx.append(torch.arange(i+300, i+350))

        if adjprc.shape[1] <= self.lookback_length:
            time_idx = torch.arange(100, 120)
            encoder_batches = features[:, 0:100, :].float()
            encoder_categorical_batches = days[:, 0:100].unsqueeze(-1).int()
            encoder_targets = x_prime[:, 0:100]

            # time_idx = torch.arange(100, adjprc.shape[-1] + self.forecast_length)
            # encoder_batches = features[:, 0:adjprc.shape[-1], :].float()
            # encoder_categorical_batches = days[:, 0:adjprc.shape[-1]].unsqueeze(-1).int()
            # encoder_targets = x_prime[:, 0:adjprc.shape[-1]]
            
            payload = {
                'encoder_cat': encoder_categorical_batches, # this should be the categorical features from the lookback period
                'encoder_cont': encoder_batches, # this should be the continous features from the lookback period
                'encoder_lengths': torch.tensor([100]), # should be length of lookback period
                'encoder_target': encoder_targets, # should be the target variable of teh lookback period (adjprc)
                'decoder_cat': torch.zeros((features.shape[0], 20, 1)).int(), # should be the categorical features of the next 50 features
                'decoder_cont': torch.zeros((features.shape[0], 20, 12)), # should be the continous features of the next 50 features
                'decoder_lengths': torch.tensor([20]), # should be the length of the prediction
                'decoder_target': torch.zeros((features.shape[0], 20)), # should be the ground truths for the next 50 adjprc
                'target_scale': torch.tensor([0, 1]).unsqueeze(0), # should be the center and scale of the robust scalar
                'decoder_time_idx': time_idx

            }
            output = self.model(payload)

            scaled_output = (output[0] * scale_.unsqueeze(-1).unsqueeze(-1)) + center_.unsqueeze(-1).unsqueeze(-1)

            return scaled_output[:, :, 3], time_idx
        else:
            outputs = []
            for b in range(adjprc.shape[0]):
                encoder_batches = []
                encoder_categorical_batches = []
                decoder_batches = []
                decoder_categorical_batches = []
                time_idx = []
                encoder_targets = []
                decoder_targets = []
                for i in range(0, adjprc.shape[1] - self.lookback_length - self.forecast_length + 1): # need to make room for prediction values
                    encoder_batches.append(features[b, i:i+self.lookback_length, :])
                    encoder_categorical_batches.append(days[b, i:i+self.lookback_length])
                    encoder_targets.append(x_prime[b, i:i+self.lookback_length])

                    decoder_batches.append(features[b, i+self.lookback_length:i+self.lookback_length+self.forecast_length, :])
                    decoder_categorical_batches.append(days[b, i+self.lookback_length : i+self.lookback_length+self.forecast_length])
                    decoder_targets.append(x_prime[b, i+self.lookback_length:i + self.lookback_length + self.forecast_length])
                    time_idx.append(torch.arange(i+self.lookback_length, i + self.lookback_length + self.forecast_length))


                encoder_batches = torch.stack(encoder_batches)
                #encoder_batches = encoder_batches.requires_grad_()
                encoder_categorical_batches = torch.stack(encoder_categorical_batches).unsqueeze(-1)

                encoder_targets = torch.stack(encoder_targets)


                decoder_batches = torch.stack(decoder_batches)
                decoder_categorical_batches = torch.stack(decoder_categorical_batches).unsqueeze(-1)

                decoder_targets = torch.stack(decoder_targets)



                payload = {
                    'encoder_cat': encoder_categorical_batches.int(), # this should be the categorical features from the lookback period
                    'encoder_cont': encoder_batches, # this should be the continous features from the lookback period
                    'encoder_lengths': torch.tensor([self.lookback_length]), # should be length of lookback period
                    'encoder_target': encoder_targets, # should be the target variable of teh lookback period (adjprc)
                    'decoder_cat': decoder_categorical_batches.int(), # should be the categorical features of the next self.forecast_length features
                    'decoder_cont': decoder_batches, # should be the continous features of the next self.forecast_length features
                    'decoder_lengths': torch.tensor([self.forecast_length]), # should be the length of the prediction
                    'decoder_target': decoder_targets, # should be the ground truths for the next self.forecast_length adjprc
                    'target_scale': torch.tensor([0, 1]).unsqueeze(0), # should be the center and scale of the robust scalar
                    'decoder_time_idx': torch.stack(time_idx)

                }
                output = self.model(payload)

                scaled_output = (output[0] * scale_[b]) + center_[b]
                outputs.append(scaled_output[:, :, 3])

                #return scaled_output[:, :, 3], time_idx

            outputs = torch.stack(outputs)
            return outputs, time_idx



    def get_days(self, recording):
        recording["date"] = pd.to_datetime(recording["date"])
        recording["adjprc_day"] = recording["date"].dt.day_of_week
        days = torch.from_numpy(recording["adjprc_day"].values) # we do not touch categorical features since it would be obvious
        days = days.type(torch.int)
        return days
    
    def get_predictions(self, outputs, time_idx):
        final_outputs = []
        time_idx = torch.stack(time_idx)
        for b in range(outputs.shape[0]):
            predictions = outputs[b].flatten()
            idx = (time_idx - time_idx[0, 0]).flatten()
            max_time = idx.max() + 1 # -300 so we start at 0
            bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=idx, src=predictions)
            bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=idx, src=torch.ones_like(predictions))
            final_predictions = bin_sums / bin_counts
            final_outputs.append(final_predictions)
        return torch.stack(final_outputs)

    def compute_gradient_penalty(self, condition: Tensor, real_data: Tensor, fake_data: Tensor):

        # random interpolation factor alpha
        alpha = torch.rand(real_data.shape[0], 1, 1, device=DEVICE)
        alpha = alpha.expand_as(real_data)

        # calculate the gradients from the interpolation output
        with torch.autograd.set_detect_anomaly(True):
            # calculate the interpolation
            interpolation = alpha * real_data + (1-alpha) * fake_data
            interpolation = interpolation.requires_grad_(True)

            # feed the interpolation into the discriminator
            interpolation_output = self.discriminator(condition, interpolation)
            grads = autograd.grad(
                outputs=interpolation_output,
                inputs=interpolation,
                grad_outputs=torch.ones_like(interpolation_output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True

            )

        gradients = grads[0]

        # flatten and compute the gradient penalty

        flattened = gradients.view(real_data.shape[0], -1)
        gradient_penalty = ((flattened.norm(2, dim=1) - 1) ** 2).mean() * self.lmda

        return gradient_penalty
    
    def training_step(self, batch):
        
        # batch would give me batches of real data
        days, real_data, real_pred = batch
        real_data = real_data.unsqueeze(-1).to(DEVICE)
        condition = real_data.clone()
        real_pred = real_pred.unsqueeze(-1).to(DEVICE)
        b, seq_len, dim = real_data.shape

        # get the optimizers for the generator and discriminator
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Step 1: we need to train the discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True
        discriminator_optimizer.zero_grad()

        discrim_real_loss = 0
        discrim_fake_loss = 0
        w_dist = 0
        g_pen = 0
        if self.current_epoch < 3:
            n_critic = self.n_critic #* 3
        else:
            n_critic = self.n_critic
        for i in range(n_critic):
            break
            # First we need to train on the real data
            discriminator_output_real = self.discriminator(condition, real_data)
            # because we classify per ticker/recording we take the average to get size b x 1
            discriminator_output_real = discriminator_output_real.mean()
            
            # Next we need to train the discriminator on the fake data
            z = torch.randn(b, real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
            fake_data = self.generator(condition, z)
            discriminator_output_fake = self.discriminator(condition, fake_data.detach())
            discriminator_output_fake = discriminator_output_fake.mean()

            # Now we have to do the gradient penalty only once per n_critic
            if i < n_critic - 3:
                gradient_penalty = self.compute_gradient_penalty(condition=condition, real_data=real_data, fake_data=fake_data)
                g_pen += gradient_penalty.detach()
                # compute the loss
                # goal is to maximize E[D(real)] - E[D(fake)], so instead we minimize E[D(fake)] - E[D(real)]
                discriminator_loss = discriminator_output_fake - discriminator_output_real + gradient_penalty
            else:
                discriminator_loss = discriminator_output_fake - discriminator_output_real
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discrim_fake_loss += discriminator_output_fake.detach()
            discrim_real_loss += discriminator_output_real.detach()
            w_dist += (discriminator_output_fake.detach() - discriminator_output_real.detach())
            g_pen += gradient_penalty.detach()
        
        print("Done Critic")

        self.discriminator_losses_fake.append(discrim_fake_loss / n_critic)
        self.discriminator_losses_real.append(discrim_real_loss / n_critic)
        self.w_dist.append(w_dist / n_critic)
        self.gradient_penalties.append(g_pen / n_critic)

        # Step 2: Train the generator
        for p in self.discriminator.parameters():
            p.requires_grad = False
            
        generator_optimizer.zero_grad()
        z = torch.randn(b, real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
        fake_data = self.generator(condition, z)
        generated_discrim_output = self.discriminator(condition, fake_data)
        generator_loss = -1 * generated_discrim_output.mean()

        self.generator_losses.append(generator_loss.detach())
        self.batch_sizes.append(b)

        x_adv = []
        days_generated = 0
        k = 0
        while days_generated < self.num_days:
            z = torch.randn(b, real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
            if days_generated == 0:
                c = condition
            else:
                c = x_adv[k-1]
            fake_data = self.generator(c, z)
            x_adv.append(fake_data)
            days_generated = days_generated + fake_data.shape[1]
            k = k + 1

        x_adv = torch.concat(x_adv, dim=1)[:, :self.num_days]


        # 1. Scale the prices back to original size
        if self.scale_max is not None and self.scale_min is not None:
            real_data_scaled = (self.scale_max - self.scale_min) * ((real_data + 1)/2) + self.scale_min
            x_adv_scaled = (self.scale_max - self.scale_min) * ((x_adv + 1)/2) + self.scale_min
            real_pred_scaled = (self.scale_max - self.scale_min) * ((real_pred + 1)/2) + self.scale_min
        else:
            real_data_scaled = real_data
            x_adv_scaled = x_adv
            real_data_scaled = real_pred


        # 2. Run model in white box setting
        #real_outputs, time_idx = self.call_model(real_data_scaled.squeeze(-1), days)
        #real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        days_pattern = torch.arange(0, 5).repeat(1, b).reshape(b, 5)
        last_day = days[:, -1].unsqueeze(-1)
        next_day = ((last_day + 1) % 5).int()
        next_week = (days_pattern + next_day) % 5

        future_days = next_week.repeat(1, self.num_days // 5)

        adv_days = torch.concat([days, future_days], dim=1)[:, :self.num_days]



        fake_outputs, time_idx = self.call_model(torch.abs(x_adv_scaled).squeeze(-1), adv_days)
        fake_predictions = self.get_predictions(fake_outputs, time_idx)


        # 3. Compute the adversarial loss (targeted)
        # slope = (fake_predictions[:, -1] - fake_predictions[:, 0]) / (self.num_days - self.lookback_length)
        direction = self.target_direction * -1
        # adversarial_loss = 10 * slope #-1 * self.c * torch.exp(direction * self.d * slope)

        x = torch.arange(self.num_days - self.lookback_length, dtype=torch.float32).expand(b, self.num_days - self.lookback_length)
        x_mean = torch.mean(x, dim=1).unsqueeze(-1)
        y_mean = torch.mean(fake_predictions, dim=1).unsqueeze(-1)

        numerator = ((x - x_mean) * (fake_predictions - y_mean)).sum(dim=1)
        denom = ((x - x_mean)**2).sum(dim=1)

        slope = numerator / denom

        adversarial_loss = slope #self.c * torch.exp(direction * self.d * slope)


        #adversarial_loss = 100 * slope
        #adversarial_loss = torch.where(slope > 0, slope, -1 * self.c * torch.exp(direction * self.d * slope))

        #adversarial_loss = torch.nn.functional.l1_loss(fake_outputs, real_pred_adjprc)
        
        # Compute the final loss and return
        adversarial_loss = -100 * torch.mean(self.beta * adversarial_loss)

        norm_loss = torch.mean(torch.norm(condition - x_adv, p=2, dim=[1]))

        total_g_loss = norm_loss + adversarial_loss
        total_g_loss.backward()
        generator_optimizer.step()
        print(adversarial_loss)

        self.step_adv_loss.append(adversarial_loss.detach())

        if self.epoch_index < len(self.epoch_betas) and self.current_epoch == self.epoch_betas[self.epoch_index]:
            self.beta = self.beta * self.beta_scale
            self.epoch_index += 1

    
    def forward(self, x):
        return self.generator(x)

    
    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2))

        return generator_optimizer, discriminator_optimizer
    


class DCGAN_Callback(pl.Callback):
    def __init__(self, real_adjprc, scaled_adjprc, days, num_to_sample, forecast_length):
        super().__init__()
        self.scaled_adjprc = scaled_adjprc
        self.real_adjprc = real_adjprc
        self.days = days
        self.num_to_sample = num_to_sample
        self.forecast_length = forecast_length


    def plot(self, values: list, x_label, y_label, title, output_file):
        
        for v in values:
            plt.plot(v)
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{y_label}')
        plt.title(f'{title}')
        plt.savefig(f'{output_file}')
        plt.close()

    def sample_adjprc(self, model, starting_point):

        real_adjprc = self.real_adjprc[starting_point : starting_point + model.num_days]
        sample_scaled_interval = self.scaled_adjprc[starting_point: starting_point + model.num_days]
        days = self.days[starting_point : starting_point + model.num_days]
        sample_forecast = self.real_adjprc[starting_point + model.num_days: starting_point + model.num_days + self.forecast_length].to(DEVICE)

        self.plot([real_adjprc.numpy()], x_label='Time (days)', y_label='Adjprc', title=f'Sample Real Adjprc: {model.num_days} Days',
                    output_file=f'{model.plot_paths}/Epoch_{model.current_epoch}/sample_real_adjprc_start={starting_point}.png')

        with torch.no_grad():
            x_adv = []
            days_generated = 0
            k = 0
            while days_generated < model.num_days:
                z = torch.randn(1, sample_scaled_interval.shape[0], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
                if days_generated == 0:
                    c = sample_scaled_interval.unsqueeze(0).unsqueeze(-1)
                else:
                    c = x_adv[k-1]
                fake_data = model.generator(c, z)
                x_adv.append(fake_data)
                days_generated = days_generated + fake_data.shape[1]
                k = k + 1

            x_adv = torch.concat(x_adv, dim=1)[:, :model.num_days]

            # 1. Scale the prices back to original size
            if model.scale_max is not None and model.scale_min is not None:
                x_adv_scaled = (model.scale_max - model.scale_min) * ((x_adv + 1)/2) + model.scale_min
            else:
                x_adv_scaled = x_adv
            
        
        plt.plot(real_adjprc.numpy(), label='Sample Real Adjprc')
        plt.plot(x_adv_scaled.squeeze(-1).squeeze(0).numpy(), label='Sample Adversarial Adjprc')
        plt.xlabel('Time (days)')
        plt.ylabel('Adjprc')
        plt.legend()
        plt.title('Sample Adversarial vs Real Adjprc Forecast')
        plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/sample_adjprc_comparison_start={starting_point}.png')
        plt.close()

        days = days.unsqueeze(0)

         # 2. Run model in white box setting
        real_outputs, time_idx = model.call_model(real_adjprc.unsqueeze(0), days)
        real_predictions = model.get_predictions(real_outputs, time_idx)
        #real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        days_pattern = torch.arange(0, 5).repeat(1, 1).reshape(1, 5)
        last_day = days[:, -1].unsqueeze(-1)
        next_day = ((last_day + 1) % 5).int()
        next_week = (days_pattern + next_day) % 5

        future_days = next_week.repeat(1, model.num_days // 5)

        adv_days = torch.concat([days, future_days], dim=1)[:, :model.num_days]



        fake_outputs, time_idx = model.call_model(x_adv_scaled.squeeze(-1), adv_days)
        fake_predictions = model.get_predictions(fake_outputs, time_idx)

        plt.plot(fake_predictions[0].detach().cpu().numpy(), label='Adversarial Pred')
        plt.plot(real_adjprc[100:].detach().cpu().numpy(), label='Actual')
        plt.plot(real_predictions[0].detach().cpu().numpy(), label='Real Pred')
        plt.xlabel('Time (days)')
        plt.ylabel('Adjprc')
        plt.legend()
        plt.title('Adversarial vs Real Adjprc Forecast')
        plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/sample_adversarial_prediction.png')
        plt.close()




    def on_train_epoch_end(self, trainer: pl.Trainer, model: AdversarialNetwork):

        generator_losses = torch.Tensor(model.generator_losses)
        discriminator_losses_real = torch.Tensor(model.discriminator_losses_real)
        discriminator_losses_fake = torch.Tensor(model.discriminator_losses_fake)
        batch_sizes = torch.Tensor(model.batch_sizes)
        w_dists = torch.Tensor(model.w_dist)
        g_pens = torch.Tensor(model.gradient_penalties)

        d_loss_real = sum(discriminator_losses_real *  batch_sizes) / sum(batch_sizes)
        d_loss_fake = sum(discriminator_losses_fake *  batch_sizes) / sum(batch_sizes)
        g_loss = sum(generator_losses *  batch_sizes) / sum(batch_sizes)
        w_dist = sum(w_dists *  batch_sizes) / sum(batch_sizes)
        g_pen = sum(g_pens * batch_sizes) / sum(batch_sizes)

        # make epoch dir
        os.makedirs(f'{model.plot_paths}/Epoch_{model.current_epoch}')
        os.chmod(f'{model.plot_paths}/Epoch_{model.current_epoch}', 0o700)
        

        # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
        num_to_sample = 128
        real_means = []
        real_stdevs = []
        real_iqr = []
        real_skew = []
        real_kurtosis = []
        conditions = torch.zeros((num_to_sample, model.num_days), device=DEVICE)
        forecasts = torch.zeros((num_to_sample, self.forecast_length), device=DEVICE)
        intervals = []
        days = []
        initial_prices = []
        for i in range(num_to_sample):
            start = random.randint(0, len(self.scaled_adjprc) - model.num_days - self.forecast_length)
            interval = self.real_adjprc[start : start + model.num_days]
            scaled_interval = self.scaled_adjprc[start: start + model.num_days]
            f = self.real_adjprc[start + model.num_days: start + model.num_days + self.forecast_length].to(DEVICE)
            forecasts[i] = f
            
            d = self.days[start: start + model.num_days]

            condition = scaled_interval.to(DEVICE)
            conditions[i] = condition
            intervals.append(interval)

            if i == 0:
                self.plot([interval.numpy()], x_label='Time (days)', y_label='Adjprc', title=f'Example Real Adjprc: {model.num_days} Days',
                          output_file=f'{model.plot_paths}/Epoch_{model.current_epoch}/example_real_adjprc_epoch{model.current_epoch}.png')
                
            # Convert to log returns for similarity score
            log_returns = torch.log(interval[1:] / interval[:-1])
                
            mean = torch.mean(log_returns)
            stdev = torch.std(log_returns)
            q75 = torch.quantile(log_returns, q=0.75)
            q25 = torch.quantile(log_returns, q=0.25)
            iqr = q75 - q25

            skew = torch.sum(((log_returns - mean) / stdev) ** 3) / model.num_days
            kurtosis = torch.sum(((log_returns - mean) / stdev) ** 4) / model.num_days

            real_means.append(mean)
            real_stdevs.append(stdev)
            real_iqr.append(iqr)
            real_skew.append(skew)
            real_kurtosis.append(kurtosis)
            days.append(d)

        starting_point = 2000
        self.sample_adjprc(model, starting_point)

        
        
        real_means = torch.stack(real_means).mean()
        real_stdevs = torch.stack(real_stdevs).mean()
        real_iqr = torch.stack(real_iqr).mean()
        real_skew = torch.stack(real_skew).mean()
        real_kurtosis = torch.stack(real_kurtosis).mean()

        conditions = conditions.unsqueeze(-1)

        real_data = conditions.clone()
        days = torch.stack(days)
       
        model.eval()
        with torch.no_grad():
            x_adv = []
            days_generated = 0
            k = 0
            while days_generated < model.num_days:
                z = torch.randn(conditions.shape[0], real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
                if days_generated == 0:
                    c = conditions
                else:
                    c = x_adv[k-1]
                fake_data = model.generator(c, z)
                x_adv.append(fake_data)
                days_generated = days_generated + fake_data.shape[1]
                k = k + 1

            x_adv = torch.concat(x_adv, dim=1)[:, :model.num_days]


            # 1. Scale the prices back to original size
            if model.scale_max is not None and model.scale_min is not None:
                real_data_scaled = (model.scale_max - model.scale_min) * ((real_data + 1)/2) + model.scale_min
                x_adv_scaled = (model.scale_max - model.scale_min) * ((x_adv + 1)/2) + model.scale_min
            else:
                real_data_scaled = real_data
                x_adv_scaled = x_adv


            self.plot([x_adv_scaled[0].numpy()], x_label='Time (days)', y_label='Adjprc', title=f'Sample Fake Adjprc: {model.num_days} Days',
                        output_file=f'{model.plot_paths}/Epoch_{model.current_epoch}/example_gan_adjprc.png')
            
        fake_output = torch.log(x_adv_scaled[:, 1:] /x_adv_scaled[:, :-1])
        mean = torch.mean(fake_output, dim=1)
        stdev = torch.std(fake_output, dim=1)
        q75 = torch.quantile(fake_output, q=0.75, dim=1)
        q25 = torch.quantile(fake_output, q=0.25, dim=1)
        iqr = q75 - q25

        z = (fake_output.squeeze(-1) - mean) / stdev

        skew = torch.mean(z ** 3, dim=1)
        kurtosis = torch.mean(z ** 4, dim=1)

        fake_means = mean.mean()
        fake_stdevs = stdev.mean()
        fake_iqr = iqr.mean()
        fake_skew = skew.mean()
        fake_kurtosis = kurtosis.mean()

        s_loss = ((1000 * (real_means - fake_means)) ** 2 + (100 * (real_stdevs - fake_stdevs)) ** 2 + 
                (real_skew - fake_skew) ** 2 + (real_kurtosis - fake_kurtosis) ** 2)

        

        # 2. Run model in white box setting
        real_outputs, time_idx = model.call_model(real_data_scaled.squeeze(-1), days)
        real_predictions = model.get_predictions(real_outputs, time_idx)
        #real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        days_pattern = torch.arange(0, 5).repeat(1, num_to_sample).reshape(num_to_sample, 5)
        last_day = days[:, -1].unsqueeze(-1)
        next_day = ((last_day + 1) % 5).int()
        next_week = (days_pattern + next_day) % 5

        future_days = next_week.repeat(1, model.num_days // 5)

        adv_days = torch.concat([days, future_days], dim=1)[:, :model.num_days]



        fake_outputs, time_idx = model.call_model(torch.abs(x_adv_scaled).squeeze(-1), adv_days)
        fake_predictions = model.get_predictions(fake_outputs, time_idx)


        # 3. Compute the adversarial loss (targeted)
        slope = (fake_predictions[:, -1] - fake_predictions[:, 0]) / (model.num_days - model.lookback_length)
        direction = model.target_direction * -1
        adversarial_loss = -1 * model.c * torch.exp(direction * model.d * slope)
        #adversarial_loss = torch.where(slope > 0, slope, -1 * model.c * torch.exp(direction * model.d * slope))

        # Compute the final loss and return
        a_loss = torch.mean(model.beta * adversarial_loss)

        x = torch.arange(model.num_days - model.lookback_length, dtype=torch.float32).expand(num_to_sample, model.num_days - model.lookback_length)
        x_mean = torch.mean(x, dim=1).unsqueeze(-1)
        y_mean = torch.mean(fake_predictions, dim=1).unsqueeze(-1)

        numerator = ((x - x_mean) * (fake_predictions - y_mean)).sum(dim=1)
        denom = ((x - x_mean)**2).sum(dim=1)

        slope = numerator / denom

        #adversarial_loss = -1 * self.c * torch.exp(direction * self.d * slope)

        adversarial_loss = slope #self.c * torch.exp(direction * self.d * slope)


        #adversarial_loss = 100 * slope
        #adversarial_loss = torch.where(slope > 0, slope, -1 * self.c * torch.exp(direction * self.d * slope))

        #adversarial_loss = torch.nn.functional.l1_loss(fake_outputs, real_pred_adjprc)
        
        # Compute the final loss and return
        adversarial_loss = -100 * torch.mean(model.beta * adversarial_loss)

        norm_loss = torch.mean(torch.norm(condition - x_adv, p=2, dim=[1]))

        f_loss = norm_loss + adversarial_loss

        #f_loss = s_loss - a_loss
        # Compute the final loss and return
        #f_loss = s_loss + a_loss




        with open(f"{model.plot_paths}/Epoch_{model.current_epoch}/generated_stats_epoch_{model.current_epoch}.txt", 'a') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Sample Mean: {real_means}, Fake Mean: {fake_means}\n")
            file.write(f"Sample Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Sample IQR: {real_iqr}, Fake IQR: {fake_iqr}\n")
            file.write(f"Sample skew: {real_skew}, Fake skew: {fake_skew}\n")
            file.write(f"Sample kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n")
            file.write(f"Similarity Loss {s_loss}\n")
            file.write(f"Adversarial Loss {a_loss}\n")
            file.write(f"Final Loss {f_loss}\n")
            file.write("=" * 50 + "\n")
        
        model.log('loss', f_loss)
        model.train()
        # plot example fake data
        # first we need to scale the fake data to match that of the log returns


        plt.plot(fake_predictions[0].detach().cpu().numpy(), label='Adversarial Pred')
        plt.plot(real_data_scaled[0][100:].detach().cpu().numpy(), label='Actual')
        plt.plot(real_predictions[0].detach().cpu().numpy(), label='Real Pred')
        plt.xlabel('Time (days)')
        plt.ylabel('Adjprc')
        plt.legend()
        plt.title('Adversarial vs Real Adjprc Forecast')
        plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/example_adversarial_prediction.png')
        plt.close()

        

        if model.current_epoch == 49:
            torch.save(model.state_dict(), f'{model.plot_paths}/mapping_gan.pt')


        print('=================================')
        print(f"Train Final Loss: {f_loss}")
        print(f"Discriminator Real Loss: {d_loss_real}")
        print(f"Discriminator Fake Loss: {d_loss_fake}")
        print(f"Generator Loss: {g_loss}")
        print(f"W Dist Approx: {w_dist}")
        print(f"Train Adversarial Loss: {a_loss}")
        print(f"Train Norm Loss: {norm_loss}")
        print('=================================')

        model.final_losses.append(f_loss)
        model.adversarial_losses.append(a_loss)

        model.step_adv_loss = []
        model.batch_sizes = []
        model.step_f_loss = []
        model.d_loss_fake.append(d_loss_fake)
        model.d_loss_real.append(d_loss_real)
        model.g_loss.append(g_loss)
        model.final_w_dists.append(w_dist)
        model.g_pens.append(g_pen)

        model.w_dist = []
        model.batch_sizes = []
        model.discriminator_losses_fake = []
        model.discriminator_losses_real = []
        model.generator_losses = []
        model.w_dist = []
        model.gradient_penalties = []
