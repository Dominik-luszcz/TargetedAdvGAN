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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AdversarialNetwork(pl.LightningModule):
    def __init__(
        self,
        sample_size,
        model,
        n_critic=5,
        target_direction=1,
        beta1=0.0,
        beta2=0.9,
        lookback_length=100,
        forecast_length=20,
        num_days=120,
        alpha=1,
        init_beta=1e-05,
        epoch_betas=[30, 60, 100, 115, 140, 190, 235],
        beta_scale=7.55,
        c=5,
        d=2,
        lmda=1,
        scale_max=None,
        scale_min=None,
        plot_paths=".",
        black_box=False,
    ):
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
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length
        self.num_days = num_days

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

        # self.save_hyperparameters()

        # Set model to eval mode (no gradient updates)
        object.__setattr__(self, "model", model)
        self.model.eval()
        # If blackbox, then set grad to false for all nhits params
        if self.black_box:
            for param in self.model.parameters():
                param.requires_grad = False

    def exponential_moving_average(self, adjprc: torch.Tensor, span: int):
        """
        Compute the ema based on the pandas formula in their documentation (when adjust=False)
        """
        alpha = 2.0 / (span + 1.0)

        ema = torch.zeros_like(adjprc)
        ema[0] = adjprc[0]

        # formula taken from the pandas definition
        for i in range(1, len(adjprc)):
            ema[i] = (1 - alpha) * adjprc[i - 1] + alpha * adjprc[i]

        return ema

    def feature_generation(self, adjprc: torch.Tensor):
        """
        Perform the feature generation for the NHITS model given the adjprc
        """
        adjprc = adjprc.float()
        weight5 = torch.ones((1, 1, 5)) / 5
        weight10 = torch.ones((1, 1, 10)) / 10
        weight20 = torch.ones((1, 1, 20)) / 20
        rolling_mean5 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1), weight5, stride=1
        )

        rolling_mean10 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1), weight10, stride=1
        )

        rolling_mean20 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1), weight20, stride=1
        )

        # stdevs: Variance = E[x^2] - E[x]^2
        # so E[x] = rolling_meanX => E[x]^2 = rolling_meanX ^ 2
        rollingExp_mean5 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1) ** 2, weight5, stride=1
        )
        rollingExp_mean10 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1) ** 2, weight10, stride=1
        )
        rollingExp_mean20 = torch.nn.functional.conv1d(
            adjprc.unsqueeze(1) ** 2, weight20, stride=1
        )

        rolling_stdev5 = torch.sqrt(
            torch.clip(rollingExp_mean5 - rolling_mean5**2 + 1e-06, min=0)
        )
        rolling_stdev10 = torch.sqrt(
            torch.clip(rollingExp_mean10 - rolling_mean10**2 + 1e-06, min=0)
        )
        rolling_stdev20 = torch.sqrt(
            torch.clip(rollingExp_mean20 - rolling_mean20**2 + 1e-06, min=0)
        )
        # but we have padding and the model fills the first x values with something (either adjprc or mean rolling_stdev) so we have to fix that
        rolling_mean5 = rolling_mean5.squeeze(1)
        rolling_mean5 = torch.cat([adjprc[:, :4], rolling_mean5], dim=1)
        rolling_stdev5 = rolling_stdev5.squeeze(1)
        rolling_stdev5 = torch.cat(
            [
                torch.mean(rolling_stdev20.squeeze(1), dim=1)
                .unsqueeze(-1)
                .repeat(1, 4),
                rolling_stdev5,
            ],
            dim=1,
        )

        rolling_mean10 = rolling_mean10.squeeze(1)
        rolling_mean10 = torch.cat([adjprc[:, :9], rolling_mean10], dim=1)
        rolling_stdev10 = rolling_stdev10.squeeze(1)
        rolling_stdev10 = torch.cat(
            [
                torch.mean(rolling_stdev20.squeeze(1), dim=1)
                .unsqueeze(-1)
                .repeat(1, 9),
                rolling_stdev10,
            ],
            dim=1,
        )

        rolling_mean20 = rolling_mean20.squeeze(1)
        rolling_mean20 = torch.cat([adjprc[:, :19], rolling_mean20], dim=1)
        rolling_stdev20 = rolling_stdev20.squeeze(1)
        rolling_stdev20 = torch.cat(
            [
                torch.mean(rolling_stdev20.squeeze(1), dim=1)
                .unsqueeze(-1)
                .repeat(1, 19),
                rolling_stdev20,
            ],
            dim=1,
        )

        # log returns
        log_returns = torch.log(adjprc[:, 1:] / adjprc[:, :-1])
        log_returns = torch.cat([torch.zeros_like(adjprc)[:, :1], log_returns], dim=1)

        # ROC (percetn change of 5)
        roc5 = (adjprc[:, 5:] - adjprc[:, :-5]) / adjprc[:, :-5]
        roc5 = torch.cat([torch.zeros_like(adjprc)[:, :5], roc5], dim=1)

        # exponential moving averages
        ema_5 = self.exponential_moving_average(adjprc=adjprc, span=5)
        ema_10 = self.exponential_moving_average(adjprc=adjprc, span=10)
        ema_20 = self.exponential_moving_average(adjprc=adjprc, span=20)

        features = torch.stack(
            [
                rolling_mean5,
                rolling_mean10,
                rolling_mean20,
                rolling_stdev5,
                rolling_stdev10,
                rolling_stdev20,
                log_returns,
                roc5,
                ema_5,
                ema_10,
                ema_20,
            ],
            dim=-1,
        )

        # Because we are doing 1 recording at a time, the standard scalar for teh features (not target) are just the mean and stdev of the recording
        # Similarly, the quantiles are the same for the target for the robust scalar for adjprc

        avg = torch.mean(features, dim=1)
        stdev = torch.std(features, dim=1)

        features = (features - avg.unsqueeze(1)) / stdev.unsqueeze(1)

        center_ = torch.median(adjprc, dim=1).values
        scale_ = (
            torch.quantile(adjprc, q=0.75, dim=1)
            - torch.quantile(adjprc, q=0.25, dim=1)
        ) / 2

        features = torch.concat([adjprc.unsqueeze(-1), features], dim=-1)

        return features, scale_, center_

    def call_model(self, adjprc: torch.Tensor, days):
        x_prime = adjprc

        features, scale_, center_ = self.feature_generation(x_prime)

        features[:, :, 0] = (
            features[:, :, 0] - center_.unsqueeze(-1)
        ) / scale_.unsqueeze(-1)

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

        time_idx = torch.arange(100, 120)
        encoder_batches = features[:, 0:100, :].float()
        encoder_categorical_batches = days[:, 0:100].unsqueeze(-1).int()
        encoder_targets = x_prime[:, 0:100]

        # will need to batch this further
        payload = {
            "encoder_cat": encoder_categorical_batches,  # this should be the categorical features from the lookback period
            "encoder_cont": encoder_batches,  # this should be the continous features from the lookback period
            "encoder_lengths": torch.tensor(
                [100]
            ),  # should be length of lookback period
            "encoder_target": encoder_targets,  # should be the target variable of teh lookback period (adjprc)
            "decoder_cat": torch.zeros(
                (features.shape[0], 20, 1)
            ).int(),  # should be the categorical features of the next 50 features
            "decoder_cont": torch.zeros(
                (features.shape[0], 20, 12)
            ),  # should be the continous features of the next 50 features
            "decoder_lengths": torch.tensor(
                [20]
            ),  # should be the length of the prediction
            "decoder_target": torch.zeros(
                (features.shape[0], 20)
            ),  # should be the ground truths for the next 50 adjprc
            "target_scale": torch.tensor([0, 1]).unsqueeze(
                0
            ),  # should be the center and scale of the robust scalar
            "decoder_time_idx": time_idx,
        }

        """
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

        """

        output = self.model(payload)

        scaled_output = (
            output[0] * scale_.unsqueeze(-1).unsqueeze(-1)
        ) + center_.unsqueeze(-1).unsqueeze(-1)

        return scaled_output[:, :, 3], time_idx

    def get_days(self, recording):
        recording["date"] = pd.to_datetime(recording["date"])
        recording["adjprc_day"] = recording["date"].dt.day_of_week
        days = torch.from_numpy(
            recording["adjprc_day"].values
        )  # we do not touch categorical features since it would be obvious
        days = days.type(torch.int)
        return days

    def get_predictions(self, outputs, time_idx):
        predictions = outputs[0][:, :, 3].flatten(1)
        time_idx = time_idx - 300
        max_time = max(time_idx) + 1  # -300 so we start at 0
        bin_sums = torch.zeros(max_time).scatter_add(
            dim=0, index=time_idx, src=predictions
        )
        bin_counts = torch.zeros(max_time).scatter_add(
            dim=0, index=time_idx, src=torch.ones_like(predictions)
        )
        final_predictions = bin_sums / bin_counts
        return final_predictions

    def compute_gradient_penalty(
        self, condition: Tensor, real_data: Tensor, fake_data: Tensor
    ):

        # random interpolation factor alpha
        alpha = torch.rand(real_data.shape[0], 1, 1, device=DEVICE)
        alpha = alpha.expand_as(real_data)

        # calculate the gradients from the interpolation output
        with torch.autograd.set_detect_anomaly(True):
            # calculate the interpolation
            interpolation = alpha * real_data + (1 - alpha) * fake_data
            interpolation = interpolation.requires_grad_(True)

            # feed the interpolation into the discriminator
            interpolation_output = self.discriminator(condition, interpolation)
            grads = autograd.grad(
                outputs=interpolation_output,
                inputs=interpolation,
                grad_outputs=torch.ones_like(interpolation_output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )

        gradients = grads[0]

        # flatten and compute the gradient penalty

        flattened = gradients.view(real_data.shape[0], -1)
        gradient_penalty = ((flattened.norm(2, dim=1) - 1) ** 2).mean() * self.lmda

        return gradient_penalty

    def training_step(self, batch):

        # batch would give me batches of real data
        initial_price, days, real_data, real_pred = batch
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
            n_critic = self.n_critic  # * 3
        else:
            n_critic = self.n_critic
        for i in range(n_critic):
            # First we need to train on the real data
            discriminator_output_real = self.discriminator(condition, real_data)
            # because we classify per ticker/recording we take the average to get size b x 1
            discriminator_output_real = discriminator_output_real.mean()

            # Next we need to train the discriminator on the fake data
            z = torch.randn(
                b, real_data.shape[1], dtype=torch.float64, device=DEVICE
            ).unsqueeze(-1)
            fake_data = self.generator(condition, z)
            discriminator_output_fake = self.discriminator(
                condition, fake_data.detach()
            )
            discriminator_output_fake = discriminator_output_fake.mean()

            # Now we have to do the gradient penalty only once per n_critic
            if i < n_critic - 2:
                gradient_penalty = self.compute_gradient_penalty(
                    condition=condition, real_data=real_data, fake_data=fake_data
                )
                g_pen += gradient_penalty.detach()
                # compute the loss
                # goal is to maximize E[D(real)] - E[D(fake)], so instead we minimize E[D(fake)] - E[D(real)]
                discriminator_loss = (
                    discriminator_output_fake
                    - discriminator_output_real
                    + gradient_penalty
                )
            else:
                discriminator_loss = (
                    discriminator_output_fake - discriminator_output_real
                )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discrim_fake_loss += discriminator_output_fake.detach()
            discrim_real_loss += discriminator_output_real.detach()
            w_dist += (
                discriminator_output_fake.detach() - discriminator_output_real.detach()
            )
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
        z = torch.randn(
            b, real_data.shape[1], dtype=torch.float64, device=DEVICE
        ).unsqueeze(-1)
        fake_data = self.generator(condition, z)
        generated_discrim_output = self.discriminator(condition, fake_data)
        generator_loss = -1 * generated_discrim_output.mean()

        self.generator_losses.append(generator_loss.detach())
        self.batch_sizes.append(b)

        # Compute mean loss
        z = torch.randn(
            b, real_data.shape[1], dtype=torch.float64, device=DEVICE
        ).unsqueeze(-1)
        fake_data = self.generator(condition, z)

        # 1. convert the log returns into adjprc
        if self.scale_max is not None and self.scale_min is not None:
            real_data_scaled = (self.scale_max - self.scale_min) * (
                (real_data + 1) / 2
            ) + self.scale_min
            x_adv_scaled = (self.scale_max - self.scale_min) * (
                (fake_data + 1) / 2
            ) + self.scale_min
            real_pred_scaled = (self.scale_max - self.scale_min) * (
                (real_pred + 1) / 2
            ) + self.scale_min
        else:
            real_data_scaled = real_data
            x_adv_scaled = fake_data
            real_data_scaled = real_pred

        adv_adjprc = torch.concat(
            [
                initial_price.unsqueeze(-1),
                initial_price.unsqueeze(-1)
                * torch.exp(torch.cumsum(x_adv_scaled.squeeze(-1), dim=1)),
            ],
            dim=1,
        )
        real_adjprc = torch.concat(
            [
                initial_price.unsqueeze(-1),
                initial_price.unsqueeze(-1)
                * torch.exp(torch.cumsum(real_data_scaled.squeeze(-1), dim=1)),
            ],
            dim=1,
        )
        real_pred_adjprc = real_adjprc[:, -1].unsqueeze(-1) * torch.exp(
            torch.cumsum(real_pred_scaled.squeeze(-1), dim=1)
        )

        # 2. Run model in white box setting
        real_outputs, time_idx = self.call_model(real_adjprc, days)
        # real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        fake_outputs, _ = self.call_model(adv_adjprc, days)
        # fake_predictions = self.get_predictions(fake_outputs, time_idx)

        # 3. Compute the adversarial loss (targeted)
        direction = self.target_direction * -1

        x = torch.arange(
            self.num_days - self.lookback_length, dtype=torch.float32
        ).expand(b, self.num_days - self.lookback_length)
        x_mean = torch.mean(x, dim=1).unsqueeze(-1)
        y_mean = torch.mean(fake_outputs, dim=1).unsqueeze(-1)

        numerator = ((x - x_mean) * (fake_outputs - y_mean)).sum(dim=1)
        denom = ((x - x_mean) ** 2).sum(dim=1)

        slope = numerator / denom

        adversarial_loss = self.c * torch.exp(direction * self.d * slope)

        # Compute the final loss and return
        adversarial_loss = torch.mean(self.beta * adversarial_loss)

        total_g_loss = generator_loss + adversarial_loss
        total_g_loss.backward()
        generator_optimizer.step()

        self.step_adv_loss.append(adversarial_loss.detach())

        if (
            self.epoch_index < len(self.epoch_betas)
            and self.current_epoch == self.epoch_betas[self.epoch_index]
        ):
            self.beta = self.beta * self.beta_scale
            self.epoch_index += 1

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2)
        )

        return generator_optimizer, discriminator_optimizer


class DCGAN_Callback(pl.Callback):
    def __init__(
        self, log_returns, scaled_returns, days, initial_prices, num_to_sample
    ):
        super().__init__()
        self.scaled_returns = scaled_returns
        self.log_returns = log_returns
        self.days = days
        self.num_to_sample = num_to_sample
        self.initial_prices = initial_prices

    def plot(self, values: list, x_label, y_label, title, output_file):

        for v in values:
            plt.plot(v)
        plt.xlabel(f"{x_label}")
        plt.ylabel(f"{y_label}")
        plt.title(f"{title}")
        plt.savefig(f"{output_file}")
        plt.close()

    def on_train_epoch_end(self, trainer: pl.Trainer, model: AdversarialNetwork):

        generator_losses = torch.Tensor(model.generator_losses)
        discriminator_losses_real = torch.Tensor(model.discriminator_losses_real)
        discriminator_losses_fake = torch.Tensor(model.discriminator_losses_fake)
        batch_sizes = torch.Tensor(model.batch_sizes)
        w_dists = torch.Tensor(model.w_dist)
        g_pens = torch.Tensor(model.gradient_penalties)

        d_loss_real = sum(discriminator_losses_real * batch_sizes) / sum(batch_sizes)
        d_loss_fake = sum(discriminator_losses_fake * batch_sizes) / sum(batch_sizes)
        g_loss = sum(generator_losses * batch_sizes) / sum(batch_sizes)
        w_dist = sum(w_dists * batch_sizes) / sum(batch_sizes)
        g_pen = sum(g_pens * batch_sizes) / sum(batch_sizes)

        # make epoch dir
        os.makedirs(f"{model.plot_paths}/Epoch_{model.current_epoch}")
        os.chmod(f"{model.plot_paths}/Epoch_{model.current_epoch}", 0o700)

        # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
        num_to_sample = 128
        real_means = []
        real_stdevs = []
        real_iqr = []
        real_skew = []
        real_kurtosis = []
        conditions = torch.zeros((num_to_sample, model.sample_size), device=DEVICE)
        forecasts = torch.zeros((num_to_sample, 20), device=DEVICE)
        intervals = []
        days = []
        initial_prices = []
        for i in range(num_to_sample):
            start = random.randint(1, len(self.log_returns) - model.sample_size - 20)
            interval = self.log_returns[start : start + model.sample_size]
            scaled_interval = self.scaled_returns[start : start + model.sample_size]
            f = self.scaled_returns[
                start + model.sample_size : start + model.sample_size + 20
            ].to(DEVICE)
            forecasts[i] = f

            d = self.days[start - 1 : start + model.sample_size]
            price = self.initial_prices[start - 1]
            condition = scaled_interval.to(DEVICE)
            conditions[i] = condition
            intervals.append(interval)
            if i == 0:
                real_adjprc = torch.concat(
                    [
                        price.unsqueeze(-1),
                        price.unsqueeze(-1) * torch.exp(torch.cumsum(interval, dim=-1)),
                    ],
                    dim=-1,
                )

                self.plot(
                    [interval.numpy()],
                    x_label="Time (days)",
                    y_label="Log Returns",
                    title=f"Example Real Returns: {model.sample_size} Days",
                    output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/example_real_log_returns_epoch{model.current_epoch}.png",
                )

                self.plot(
                    [real_adjprc.numpy()],
                    x_label="Time (days)",
                    y_label="Adjprc",
                    title=f"Example Real Adjprc: {model.sample_size} Days",
                    output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/example_real_adjprc_epoch{model.current_epoch}.png",
                )

            mean = torch.mean(interval)
            stdev = torch.std(interval)
            q75 = torch.quantile(interval, q=0.75)
            q25 = torch.quantile(interval, q=0.25)
            iqr = q75 - q25

            skew = torch.sum(((interval - mean) / stdev) ** 3) / model.sample_size
            kurtosis = torch.sum(((interval - mean) / stdev) ** 4) / model.sample_size

            real_means.append(mean)
            real_stdevs.append(stdev)
            real_iqr.append(iqr)
            real_skew.append(skew)
            real_kurtosis.append(kurtosis)
            days.append(d)
            initial_prices.append(price)

        starting_point = 1000

        sample_interval = self.log_returns[
            starting_point : starting_point + model.sample_size
        ]
        sample_scaled_interval = self.scaled_returns[
            starting_point : starting_point + model.sample_size
        ]
        sample_price = self.initial_prices[starting_point - 1]
        sample_days = self.days[starting_point - 1 : starting_point + model.sample_size]
        sample_forecast = self.scaled_returns[
            starting_point
            - 1
            + model.sample_size : starting_point
            - 1
            + model.sample_size
            + 20
        ].to(DEVICE)

        sample_real_adjprc = torch.concat(
            [
                sample_price.unsqueeze(-1),
                sample_price.unsqueeze(-1)
                * torch.exp(torch.cumsum(sample_interval, dim=-1)),
            ],
            dim=-1,
        )

        self.plot(
            [sample_interval.numpy()],
            x_label="Time (days)",
            y_label="Log Returns",
            title=f"Sample Real Returns: {model.sample_size} Days",
            output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/REAL_sample_log_returns_start={starting_point}.png",
        )

        self.plot(
            [sample_real_adjprc.numpy()],
            x_label="Time (days)",
            y_label="Adjprc",
            title=f"Sample Real Adjprc: {model.sample_size} Days",
            output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/REAL_sample_interval_adjprc_start={starting_point}.png",
        )

        real_means = torch.stack(real_means).mean()
        real_stdevs = torch.stack(real_stdevs).mean()
        real_iqr = torch.stack(real_iqr).mean()
        real_skew = torch.stack(real_skew).mean()
        real_kurtosis = torch.stack(real_kurtosis).mean()

        conditions = conditions.unsqueeze(-1)

        model.eval()
        with torch.no_grad():
            z = torch.randn(
                num_to_sample, model.sample_size, dtype=torch.float64, device=DEVICE
            ).unsqueeze(-1)
            fake_output = model.generator(conditions, z)

            # Need to scale back to log returns
            if model.scale_max != None and model.scale_min != None:

                fake_output = (model.scale_max - model.scale_min) * (
                    (fake_output + 1) / 2
                ) + model.scale_min

            model.example_outputs.append(fake_output)

            sample_z = torch.randn(
                1, model.sample_size, dtype=torch.float64, device=DEVICE
            ).unsqueeze(-1)
            fake_sample_interval = model.generator(
                sample_scaled_interval.unsqueeze(0).unsqueeze(-1), sample_z
            )

            # Need to scale back to log returns
            if model.scale_max != None and model.scale_min != None:

                fake_sample_interval = (model.scale_max - model.scale_min) * (
                    (fake_sample_interval + 1) / 2
                ) + model.scale_min

            sample_adv_adjprc = torch.concat(
                [
                    sample_price.unsqueeze(-1),
                    sample_price.unsqueeze(-1)
                    * torch.exp(
                        torch.cumsum(
                            fake_sample_interval.squeeze(-1).squeeze(0), dim=-1
                        )
                    ),
                ],
                dim=-1,
            )

            self.plot(
                [fake_sample_interval.squeeze(-1).squeeze(0).detach().cpu().numpy()],
                x_label="Time (days)",
                y_label="Log Returns",
                title=f"Sample Fake Returns: {model.sample_size} Days",
                output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/FAKE_sample_log_returns_start={starting_point}.png",
            )

            self.plot(
                [sample_adv_adjprc.numpy()],
                x_label="Time (days)",
                y_label="Adjprc",
                title=f"Sample Fake Adjprc: {model.sample_size} Days",
                output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/FAKE_sample_interval_adjprc_start={starting_point}.png",
            )

        plt.plot(sample_real_adjprc.numpy(), label="Sample Real Adjprc")
        plt.plot(sample_adv_adjprc.numpy(), label="Sample Adversarial Adjprc")
        plt.xlabel("Time (days)")
        plt.ylabel("Adjprc")
        plt.legend()
        plt.title("Sample Adversarial vs Real Adjprc Forecast")
        plt.savefig(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/sample_adjprc_comparison_start={starting_point}.png"
        )
        plt.close()

        mean = torch.mean(fake_output, dim=1)
        stdev = torch.std(fake_output, dim=1)
        q75 = torch.quantile(fake_output, q=0.75, dim=1)
        q25 = torch.quantile(fake_output, q=0.25, dim=1)
        iqr = q75 - q25

        z = (fake_output.squeeze(-1) - mean) / stdev

        skew = torch.mean(z**3, dim=1)
        kurtosis = torch.mean(z**4, dim=1)

        fake_means = mean.mean()
        fake_stdevs = stdev.mean()
        fake_iqr = iqr.mean()
        fake_skew = skew.mean()
        fake_kurtosis = kurtosis.mean()

        s_loss = (
            (1000 * (real_means - fake_means)) ** 2
            + (100 * (real_stdevs - fake_stdevs)) ** 2
            + (real_skew - fake_skew) ** 2
            + (real_kurtosis - fake_kurtosis) ** 2
        )

        # Now we have to do the adversarial attack

        real_data = conditions.clone()
        fake_data = fake_output
        real_pred = forecasts.clone()
        initial_price = torch.stack(initial_prices)
        days = torch.stack(days)

        # 1. convert the log returns into adjprc
        if model.scale_max is not None and model.scale_min is not None:
            real_data_scaled = (model.scale_max - model.scale_min) * (
                (real_data + 1) / 2
            ) + model.scale_min
            x_adv_scaled = fake_data
            real_pred_scaled = (model.scale_max - model.scale_min) * (
                (real_pred + 1) / 2
            ) + model.scale_min
        else:
            real_data_scaled = real_data
            x_adv_scaled = fake_data
            real_pred_scaled = real_pred

        adv_adjprc = torch.concat(
            [
                initial_price.unsqueeze(-1),
                initial_price.unsqueeze(-1)
                * torch.exp(torch.cumsum(x_adv_scaled.squeeze(-1), dim=1)),
            ],
            dim=1,
        )
        real_adjprc = torch.concat(
            [
                initial_price.unsqueeze(-1),
                initial_price.unsqueeze(-1)
                * torch.exp(torch.cumsum(real_data_scaled.squeeze(-1), dim=1)),
            ],
            dim=1,
        )
        real_pred_adjprc = real_adjprc[:, -1].unsqueeze(-1) * torch.exp(
            torch.cumsum(real_pred_scaled.squeeze(-1), dim=1)
        )

        if model.current_epoch > 20:
            print("here")

        # 2. Run model in white box setting
        real_outputs, time_idx = model.call_model(real_adjprc, days)

        fake_outputs, _ = model.call_model(adv_adjprc, days)

        # 3. Compute the adversarial loss
        direction = model.target_direction * -1

        x = torch.arange(
            model.num_days - model.lookback_length, dtype=torch.float32
        ).expand(num_to_sample, model.num_days - model.lookback_length)
        x_mean = torch.mean(x, dim=1).unsqueeze(-1)
        y_mean = torch.mean(fake_outputs, dim=1).unsqueeze(-1)

        numerator = ((x - x_mean) * (fake_outputs - y_mean)).sum(dim=1)
        denom = ((x - x_mean) ** 2).sum(dim=1)

        slope = numerator / denom

        adversarial_loss = model.c * torch.exp(direction * model.d * slope)
        a_loss = torch.mean(model.beta * adversarial_loss)

        f_loss = s_loss + a_loss

        with open(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/generated_stats_epoch_{model.current_epoch}.txt",
            "a",
        ) as file:
            file.write("=" * 50 + "\n")
            file.write(f"Sample Mean: {real_means}, Fake Mean: {fake_means}\n")
            file.write(f"Sample Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Sample IQR: {real_iqr}, Fake IQR: {fake_iqr}\n")
            file.write(f"Sample skew: {real_skew}, Fake skew: {fake_skew}\n")
            file.write(
                f"Sample kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n"
            )
            file.write(f"Similarity Loss {s_loss}\n")
            file.write(f"Adversarial Loss {a_loss}\n")
            file.write(f"Final Loss {f_loss}\n")
            file.write("=" * 50 + "\n")

        model.log("loss", f_loss)
        model.train()
        # plot example fake data
        # first we need to scale the fake data to match that of the log returns

        self.plot(
            [fake_output.squeeze(-1).squeeze(0).detach()[0].cpu()],
            x_label="Time (days)",
            y_label="Log Returns",
            title=f"Example Log Return Created From GAN: {model.sample_size} Days",
            output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/example_gan_output_log_return.png",
        )

        self.plot(
            [adv_adjprc.squeeze(-1).squeeze(0).detach()[0].cpu()],
            x_label="Time (days)",
            y_label="Adjprc",
            title=f"Example Adjprc Created From GAN: {model.sample_size} Days",
            output_file=f"{model.plot_paths}/Epoch_{model.current_epoch}/example_gan_output_adjprc.png",
        )

        fake_output = fake_outputs[0].detach().cpu()
        plt.plot(fake_output.numpy(), label="Adversarial Pred")
        plt.plot(real_pred_adjprc[0].detach().cpu().numpy(), label="Actual")
        plt.plot(real_outputs[0].detach().cpu().numpy(), label="Real Pred")
        plt.xlabel("Time (days)")
        plt.ylabel("Adjprc")
        plt.legend()
        plt.title("Adversarial vs Real Adjprc Forecast")
        plt.savefig(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/adversarial_prediction.png"
        )
        plt.close()

        # 2. Run model in white box setting
        sample_real_outputs, _ = model.call_model(
            sample_real_adjprc.unsqueeze(0), sample_days.unsqueeze(0)
        )
        # real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

        sample_fake_outputs, _ = model.call_model(
            sample_adv_adjprc.unsqueeze(0), sample_days.unsqueeze(0)
        )

        sample_real_pred = (model.scale_max - model.scale_min) * (
            (sample_forecast + 1) / 2
        ) + model.scale_min
        sample_real_forecasts = sample_real_adjprc[-1].unsqueeze(-1) * torch.exp(
            torch.cumsum(sample_real_pred, dim=-1)
        )

        plt.plot(
            sample_fake_outputs.squeeze(0).detach().cpu().numpy(),
            label="Adversarial Pred",
        )
        plt.plot(sample_real_forecasts.detach().cpu().numpy(), label="Actual")
        plt.plot(
            sample_real_outputs.squeeze(0).detach().cpu().numpy(), label="Real Pred"
        )
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Adjprc")
        plt.title("Sample Adversarial vs Real Adjprc Forecast")
        plt.savefig(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/sample_interval_forecasts.png"
        )
        plt.close()

        if model.current_epoch == 60:
            torch.save(model.state_dict(), f"{model.plot_paths}/mapping_gan.pt")

        print("=================================")
        print(f"Train Final Loss: {f_loss}")
        print(f"Discriminator Real Loss: {d_loss_real}")
        print(f"Discriminator Fake Loss: {d_loss_fake}")
        print(f"Generator Loss: {g_loss}")
        print(f"W Dist Approx: {w_dist}")
        print(f"Train Adversarial Loss: {a_loss}")
        print(f"Train Similarity Loss: {s_loss}")
        print("=================================")

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
