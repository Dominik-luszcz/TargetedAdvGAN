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
    def __init__(self, sample_size, time_series_metrics: dict, model,
                 alpha = 2, init_beta = 0.00001, epoch_betas = [100, 300, 400, 450], beta_scale = 5, 
                 scale_max = None, scale_min = None, plot_paths = '.'):
        super().__init__()

        self.sample_size = sample_size
        self.time_series_metrics = time_series_metrics
        self.model = model
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.alpha = alpha
        self.beta = init_beta
        self.plot_paths = plot_paths
        self.epoch_betas = epoch_betas
        self.epoch_index = 0
        self.beta_scale = beta_scale

        self.generator = Generator(output_dim=1, sample_size=self.sample_size)

        self.step_sim_loss = []
        self.step_adv_loss = []
        self.step_f_loss = []

        self.batch_sizes = []

        self.final_losses = []
        self.adversarial_losses = []
        self.similarity_losses = []

        self.model.eval()


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

        # encoder_batches = []
        # encoder_categorical_batches = []
        # decoder_batches = []
        # decoder_categorical_batches = []
        # time_idx = []
        # encoder_targets = []
        # decoder_targets = []

        # for i in range(0, features.shape[1] - 300 - 49): # need to make room for prediction values
        #     encoder_batches.append(features[:, i:i+300, :])
        #     encoder_categorical_batches.append(days[:, i:i+300])
        #     encoder_targets.append(x_prime[:, i:i+300])

        #     decoder_batches.append(features[:, i+300:i+300+50, :])
        #     decoder_categorical_batches.append(days[:, i+300 : i+300+50])
        #     decoder_targets.append(x_prime[:, i+300:i+350])
        #     time_idx.append(torch.arange(i+300, i+350))

        time_idx = torch.arange(100, 120)
        encoder_batches = features[:, 0:100, :].float()
        encoder_categorical_batches = days[:, 0:100].unsqueeze(-1).int()
        encoder_targets = x_prime[:, 0:100]


        decoder_batches = features[:, 100:120, :]
        decoder_categorical_batches = days[:, 100 : 120].unsqueeze(-1).int()
        decoder_targets = x_prime[:, 100:120]
        
        # will need to batch this further
        payload = {
            'encoder_cat': encoder_categorical_batches, # this should be the categorical features from the lookback period
            'encoder_cont': encoder_batches, # this should be the continous features from the lookback period
            'encoder_lengths': torch.tensor([100]), # should be length of lookback period
            'encoder_target': encoder_targets, # should be the target variable of teh lookback period (adjprc)
            'decoder_cat': decoder_categorical_batches, # should be the categorical features of the next 50 features
            'decoder_cont': decoder_batches, # should be the continous features of the next 50 features
            'decoder_lengths': torch.tensor([20]), # should be the length of the prediction
            'decoder_target': decoder_targets, # should be the ground truths for the next 50 adjprc
            'target_scale': torch.tensor([0, 1]).unsqueeze(0), # should be the center and scale of the robust scalar
            'decoder_time_idx': time_idx

        }

        '''
        output[0] = tensor of shape [2, 50, 7] # 7 because we have 7 quantiles, so 2 batches, 50 forward proj, 7 quantiles, quantile 0.5 is the prediction so it would be index 3 for the prediction
        output[1] = tensor of shape [2, 300, 1] # 2 batches, 300 for the lookback
        output[2] = tuple of length 4: 
            output[2][0]: tensor of shape [2, 300, 1] # forecasts
            output[2][1]: tensor of shape [2, 300, 1]
            output[2][2]: tensor of shape [2, 300, 1]
            output[2][3]: tensor of shape [2, 300, 1]
        output[3] = tuple of length 4:
            output[3][0]: tensor of shape [2, 50, 7] # backcasts
            output[3][1]: tensor of shape [2, 50, 7]
            output[3][2]: tensor of shape [2, 50, 7]
            output[3][3]: tensor of shape [2, 50, 7]

        '''
        
        output = self.model(payload)

        scaled_output = (output[0] * scale_.unsqueeze(-1).unsqueeze(-1)) + center_.unsqueeze(-1).unsqueeze(-1)

        return scaled_output[:, :, 3], time_idx


    def get_days(self, recording):
        recording["date"] = pd.to_datetime(recording["date"])
        recording["adjprc_day"] = recording["date"].dt.day_of_week
        days = torch.from_numpy(recording["adjprc_day"].values) # we do not touch categorical features since it would be obvious
        days = days.type(torch.int)
        return days
    
    def get_predictions(self, outputs, time_idx):
        predictions = outputs[0][:, :, 3].flatten(1)
        time_idx = time_idx - 300
        max_time = max(time_idx) + 1 # -300 so we start at 0
        bin_sums = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=predictions)
        bin_counts = torch.zeros(max_time).scatter_add(dim=0, index=time_idx, src=torch.ones_like(predictions))
        final_predictions = bin_sums / bin_counts
        return final_predictions


    
    def training_step(self, batch):
        
        # batch would give me batches of real data
        initial_price, days, real_data = batch
        real_data = real_data.unsqueeze(-1).to(DEVICE)
        b, seq_len, dim = real_data.shape

        z = torch.randn(b, real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)

        x_adv = self.forward(z)

        # First we have to make sure that the data is realistic
        if self.scale_max is not None and self.scale_min is not None:
            x_adv_scaled = (self.scale_max - self.scale_min) * ((x_adv + 1)/2) + self.scale_min
        else:
            x_adv_scaled = x_adv

        x_adv_mean = torch.mean(x_adv_scaled, dim=1)
        mean_loss = (x_adv_mean - self.time_series_metrics["mean"])**2

        x_adv_std = torch.std(x_adv_scaled, dim=1)
        std_loss = (x_adv_std - self.time_series_metrics["stdev"])**2

        x_adv_scaled = x_adv_scaled.squeeze(-1)
        z_score = (x_adv_scaled - x_adv_mean) / x_adv_std
        x_adv_skew =torch.mean(z_score ** 3, dim=1)
        skew_loss = (x_adv_skew - self.time_series_metrics["skew"])**2 
           
        x_adv_kurtosis = torch.mean(z_score ** 4, dim=1) - 3
        kurtosis_loss = (x_adv_kurtosis - self.time_series_metrics["kurtosis"])**2 
        
        similarity_loss = torch.mean(mean_loss + std_loss + skew_loss + kurtosis_loss)

        # Now we have to get an adversarial loss

        # 1. convert the log returns into adjprc
        if self.scale_max is not None and self.scale_min is not None:
            real_data_scaled = (self.scale_max - self.scale_min) * ((real_data + 1)/2) + self.scale_min
        else:
            real_data_scaled = real_data
        adv_adjprc = torch.concat([initial_price.unsqueeze(-1), initial_price.unsqueeze(-1) * torch.exp(torch.cumsum(x_adv_scaled, dim=1))], dim=1)
        real_adjprc = torch.concat([initial_price.unsqueeze(-1), initial_price.unsqueeze(-1) * torch.exp(torch.cumsum(real_data_scaled.squeeze(-1), dim=1))], dim=1)

        # 2. Run model in white box setting
        real_outputs, time_idx = self.call_model(real_adjprc, days)
        #real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        fake_outputs, _ = self.call_model(adv_adjprc, days)
        #fake_predictions = self.get_predictions(fake_outputs, time_idx)

        # 3. Compute the adversarial loss
        adversarial_loss = torch.nn.functional.l1_loss(fake_outputs, real_outputs)
        
        # Compute the final loss and return
        final_loss = self.alpha * similarity_loss - self.beta * adversarial_loss

        self.step_f_loss.append(final_loss.detach())
        self.step_adv_loss.append(adversarial_loss.detach())
        self.step_sim_loss.append(similarity_loss.detach())
        self.batch_sizes.append(b)

        if self.epoch_index < len(self.epoch_betas) and self.current_epoch == self.epoch_betas[self.epoch_index]:
            self.beta = self.beta * self.beta_scale
            self.epoch_index += 1

        return final_loss
    
    def forward(self, z):
        return self.generator(z)
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-05, weight_decay=1e-05)
    


class DCGAN_Callback(pl.Callback):
    def __init__(self, log_returns, days, initial_prices, num_to_sample):
        super().__init__()
        self.log_returns = log_returns
        self.days = days
        self.num_to_sample = num_to_sample
        self.initial_prices = initial_prices

    def on_train_epoch_end(self, trainer: pl.Trainer, model: AdversarialNetwork):

        final_losses = torch.Tensor(model.step_f_loss)
        adversarial_losses = torch.Tensor(model.step_adv_loss)
        similarity_losses = torch.Tensor(model.step_sim_loss)
        batch_sizes = torch.Tensor(model.batch_sizes)

        adversarial_loss = sum(adversarial_losses *  batch_sizes) / sum(batch_sizes)
        similarity_loss = sum(similarity_losses *  batch_sizes) / sum(batch_sizes)
        final_loss = sum(final_losses *  batch_sizes) / sum(batch_sizes)


        # make epoch dir
        os.makedirs(f'{model.plot_paths}/Epoch_{model.current_epoch}')
        os.chmod(f'{model.plot_paths}/Epoch_{model.current_epoch}', 0o700)
        

        # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
        num_to_sample = self.num_to_sample
        intervals = []
        days = []
        initial_prices = []
        for i in range(num_to_sample):
            start = random.randint(1, len(self.log_returns) - model.sample_size)
            interval = self.log_returns[start : start + model.sample_size]
            d = self.days[start - 1 : start + model.sample_size]
            price = self.initial_prices[start - 1]
            if i == 0:
                plt.plot(interval.numpy())
                plt.xlabel('Time (days)')
                plt.ylabel('Log Returns')
                plt.title(f'Example real Return: {model.sample_size} Days ')
                plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/example_real_output_epoch{model.current_epoch}.png')
                plt.close()

            intervals.append(interval)
            days.append(d)
            initial_prices.append(price)

        
       
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_to_sample, model.sample_size, dtype=torch.float64, device=DEVICE).unsqueeze(-1)
            fake_output = model.forward(z)

            # Need to scale back to log returns
            if model.scale_max != None and model.scale_min != None:

                fake_output = (model.scale_max - model.scale_min) * ((fake_output + 1)/2) + model.scale_min

            #model.example_outputs.append(fake_output)

        mean = torch.mean(fake_output, dim=1)
        stdev = torch.std(fake_output, dim=1)

        z = (fake_output.squeeze(-1) - mean) / stdev

        skew = torch.mean(z ** 3, dim=1)
        kurtosis = torch.mean(z ** 4, dim=1)

        fake_means = mean.mean()
        fake_stdevs = stdev.mean()
        fake_skew = skew.mean()
        fake_kurtosis = kurtosis.mean()

        s_loss = ((model.time_series_metrics["mean"] - fake_means) ** 2 + (model.time_series_metrics["stdev"] - fake_stdevs) ** 2 + 
                (model.time_series_metrics["skew"] - fake_skew) ** 2 + (model.time_series_metrics["kurtosis"] - fake_kurtosis) ** 2)



        # Now we have to do the adversarial attack

        # 1. convert the log returns into adjprc
        initial_price = torch.tensor(initial_prices)
        real_data = torch.stack(intervals)
        days = torch.stack(days)

        if model.scale_max is not None and model.scale_min is not None:
            real_data_scaled = (model.scale_max - model.scale_min) * ((real_data + 1)/2) + model.scale_min
        else:
            real_data_scaled = real_data
        adv_adjprc = torch.concat([initial_price.unsqueeze(-1), initial_price.unsqueeze(-1) * torch.exp(torch.cumsum(fake_output, dim=1).squeeze(-1))], dim=1)
        real_adjprc = torch.concat([initial_price.unsqueeze(-1), initial_price.unsqueeze(-1) * torch.exp(torch.cumsum(real_data_scaled, dim=1))], dim=1)

        # 2. Run model in white box setting
        real_outputs, time_idx = model.call_model(real_adjprc, days)
        #real_predictions = model.get_predictions(real_outputs, time_idx)

        fake_outputs, _ = model.call_model(adv_adjprc, days)
        #fake_predictions = model.get_predictions(fake_outputs, time_idx)

        # 3. Compute the adversarial loss
        a_loss = torch.nn.functional.l1_loss(fake_outputs, real_outputs)
        
        # Compute the final loss and return
        f_loss = model.alpha * s_loss + model.beta * a_loss




        with open(f"{model.plot_paths}/Epoch_{model.current_epoch}/generated_stats_epoch_{model.current_epoch}.txt", 'a') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Global Mean: {model.time_series_metrics["mean"]}, Fake Mean: {fake_means}\n")
            file.write(f"Global Stdev: {model.time_series_metrics["stdev"]}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Global skew: {model.time_series_metrics["skew"]}, Fake skew: {fake_skew}\n")
            file.write(f"Global kurtosis: {model.time_series_metrics["kurtosis"]}, Fake kurtosis: {fake_kurtosis}\n")
            file.write(f"Similarity Loss {s_loss}\n")
            file.write(f"Adversarial Loss {a_loss}\n")
            file.write(f"Final Loss {f_loss}\n")
            file.write("=" * 50 + "\n")

        
        
        
        
        
        model.log('loss', f_loss)
        model.train()
        # plot example fake data
        # first we need to scale the fake data to match that of the log returns
        fake_output = fake_output.squeeze(-1).squeeze(0).detach()[0].cpu()
        plt.plot(fake_output.numpy())
        plt.xlabel('Time (days)')
        plt.ylabel('Log Returns')
        plt.title('Example log return created from GAN')
        plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/example_gan_output_epoch{model.current_epoch}.png')
        plt.close()


        print('=================================')
        print(f"Train Final Loss: {final_loss}")
        print(f"Train Adversarial Loss: {adversarial_loss}")
        print(f"Train Similarity Loss: {similarity_loss}")
        print('=================================')

        model.final_losses.append(final_loss)
        model.similarity_losses.append(similarity_loss)
        model.adversarial_losses.append(adversarial_loss)

        model.step_adv_loss = []
        model.batch_sizes = []
        model.step_f_loss = []
        model.step_sim_loss = []
