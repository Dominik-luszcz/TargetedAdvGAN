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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        generator_hidden_dim,
        noise_dim,
        generator_output_dim,
        discriminator_hidden_dim,
        sample_size,
        scale_max=None,
        scale_min=None,
        lr=0.0001,
        plot_paths=".",
    ):
        super().__init__()

        self.lr = lr

        self.generator_hidden_dim = generator_hidden_dim
        self.noise_dim = noise_dim
        self.output_dim = generator_output_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.plot_paths = plot_paths
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.sample_size = sample_size

        self.generator = Generator(
            output_dim=generator_output_dim, sample_size=sample_size
        )
        self.discriminator = Discriminator(
            input_dim=1, hidden_dim=discriminator_hidden_dim, output_dim=1
        )

        self.automatic_optimization = False

        self.discriminator_losses_real = []
        self.discriminator_losses_fake = []
        self.generator_losses = []
        self.batch_sizes = []
        self.w_dist = []
        self.example_outputs = []

        self.d_loss_real = []
        self.d_loss_fake = []
        self.g_loss = []
        self.final_w_dists = []

        self.save_hyperparameters()

    def training_step(self, batch):
        # get the optimizers for the generator and discriminator
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # batch would give me batches of real data
        real_data = batch
        real_data = real_data.unsqueeze(-1).to(DEVICE)
        b, seq_len, dim = real_data.shape
        real_data_labels = torch.ones((b, 1), dtype=torch.float64, device=DEVICE) - 0.15
        fake_data_labels = (
            torch.zeros((b, 1), dtype=torch.float64, device=DEVICE) + 0.15
        )

        # Step 1: we need to train the discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True
        discriminator_optimizer.zero_grad()

        # First we need to train on the real data
        discriminator_output_real = self.discriminator(real_data)
        # because we classify per ticker/recording we take the average to get size b x 1
        discriminator_output_real = discriminator_output_real.mean(dim=1)
        discriminator_loss_real = nn.functional.binary_cross_entropy(
            discriminator_output_real, real_data_labels
        )
        discriminator_loss_real.backward()

        # Next we need to train the discriminator on the fake data
        z = torch.randn(
            b, self.noise_dim, dtype=torch.float64, device=DEVICE
        ).unsqueeze(-1)
        fake_data = self.generator(z)
        discriminator_output_fake = self.discriminator(fake_data.detach())
        discriminator_output_fake = discriminator_output_fake.mean(dim=1)
        discriminator_loss_fake = nn.functional.binary_cross_entropy(
            discriminator_output_fake, fake_data_labels
        )
        discriminator_loss_fake.backward()

        self.discriminator_losses_real.append(discriminator_loss_real.detach())
        self.discriminator_losses_fake.append(discriminator_loss_fake.detach())

        discriminator_optimizer.step()

        # Step 2: Train the generator
        for p in self.discriminator.parameters():
            p.requires_grad = False

        generator_optimizer.zero_grad()
        z = torch.randn(
            b, self.noise_dim, dtype=torch.float64, device=DEVICE
        ).unsqueeze(-1)
        fake_data = self.generator(z)
        discriminator_output_fake = self.discriminator(fake_data)
        discriminator_output_fake = discriminator_output_fake.mean(dim=1)

        generator_loss = nn.functional.binary_cross_entropy(
            discriminator_output_fake, real_data_labels + 0.15
        )

        generator_loss.backward()

        # generator_grad = 0
        # for param in self.generator.parameters():
        #     if param.grad is not None:
        #         generator_grad += param.grad.abs().sum().item()

        # print(generator_grad)
        # fake_data = self.generator(z)

        # mean = torch.mean(fake_data, dim=1)
        # stdev = torch.std(fake_data, dim=1)
        # kurtosis = torch.sum(((fake_data - mean[:, None]) / stdev[:, None]) ** 4, dim=1).mean(dim=1)
        # fake_kurtosis = kurtosis.mean()

        # mean = torch.mean(real_data, dim=1)
        # stdev = torch.std(real_data, dim=1)
        # kurtosis = torch.sum(((real_data - mean[:, None]) / stdev[:, None]) ** 4, dim=1).mean(dim=1)
        # real_kurtosis = kurtosis.mean()

        # k_loss = torch.abs(real_kurtosis - fake_kurtosis)
        # k_loss.backward()

        # Maybe try a sort of stdev loss

        # stdev_loss = fake_stdev / (real_stdev + 1e-06) + real_stdev / (real_stdev + 1e-06)
        # stdev_loss.backward()
        generator_optimizer.step()

        self.generator_losses.append(generator_loss.detach())
        self.batch_sizes.append(b)

        # self.log("wass_dist", wass_dist, on_step=False, on_epoch=True, batch_size=b)

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1e-03, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-05, betas=(0.5, 0.999)
        )

        return generator_optimizer, discriminator_optimizer


class DCGAN_Callback(pl.Callback):
    def __init__(self, log_returns):
        super().__init__()
        self.log_returns = log_returns

    def on_train_epoch_end(self, trainer: pl.Trainer, model: DCGAN):

        generator_losses = torch.Tensor(model.generator_losses)
        discriminator_losses_real = torch.Tensor(model.discriminator_losses_real)
        discriminator_losses_fake = torch.Tensor(model.discriminator_losses_fake)
        batch_sizes = torch.Tensor(model.batch_sizes)

        d_loss_real = sum(discriminator_losses_real * batch_sizes) / sum(batch_sizes)
        d_loss_fake = sum(discriminator_losses_fake * batch_sizes) / sum(batch_sizes)
        g_loss = sum(generator_losses * batch_sizes) / sum(batch_sizes)

        # make epoch dir
        os.makedirs(f"{model.plot_paths}/Epoch_{model.current_epoch}")
        os.chmod(f"{model.plot_paths}/Epoch_{model.current_epoch}", 0o700)

        # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
        num_to_sample = 16
        real_means = []
        real_stdevs = []
        real_iqr = []
        real_skew = []
        real_kurtosis = []
        for i in range(num_to_sample):
            start = random.randint(0, len(self.log_returns) - model.sample_size)
            interval = self.log_returns[start : start + model.sample_size]
            if i == 0:
                plt.plot(interval.numpy())
                plt.xlabel("Time (days)")
                plt.ylabel("Log Returns")
                plt.title(f"Example real Return: {model.sample_size} Days ")
                plt.savefig(
                    f"{model.plot_paths}/Epoch_{model.current_epoch}/example_real_output_epoch{model.current_epoch}.png"
                )
                plt.close()

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

        real_means = torch.stack(real_means).mean()
        real_stdevs = torch.stack(real_stdevs).mean()
        real_iqr = torch.stack(real_iqr).mean()
        real_skew = torch.stack(real_skew).mean()
        real_kurtosis = torch.stack(real_kurtosis).mean()

        model.eval()
        with torch.no_grad():
            z = torch.randn(
                num_to_sample, 50, dtype=torch.float64, device=DEVICE
            ).unsqueeze(-1)
            fake_output = model.generator(z)

            # Need to scale back to log returns
            if model.scale_max != None and model.scale_min != None:

                fake_output = (model.scale_max - model.scale_min) * (
                    (fake_output + 1) / 2
                ) + model.scale_min

            model.example_outputs.append(fake_output)

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

        loss = (
            (real_means - fake_means) ** 2
            + (real_stdevs - fake_stdevs) ** 2
            + (real_skew - fake_skew) ** 2
            + (real_kurtosis - fake_kurtosis) ** 2
        )

        with open(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/generated_stats_epoch_{model.current_epoch}.txt",
            "a",
        ) as file:
            file.write("=" * 50 + "\n")
            file.write(f"Real Mean: {real_means}, Fake Mean: {fake_means}\n")
            file.write(f"Real Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Real iqr: {real_iqr}, Fake iqr: {fake_iqr}\n")
            file.write(f"Real skew: {real_skew}, Fake skew: {fake_skew}\n")
            file.write(
                f"Real kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n"
            )
            file.write(f"Loss {loss}\n")
            file.write("=" * 50 + "\n")

        model.log("loss", loss)
        model.train()
        # plot example fake data
        # first we need to scale the fake data to match that of the log returns
        fake_output = fake_output.squeeze(-1).squeeze(0).detach()[0].cpu()
        plt.plot(fake_output.numpy())
        plt.xlabel("Time (days)")
        plt.ylabel("Log Returns")
        plt.title("Example log return created from GAN")
        plt.savefig(
            f"{model.plot_paths}/Epoch_{model.current_epoch}/example_gan_output_epoch{model.current_epoch}.png"
        )
        plt.close()

        print("=================================")
        print(f"Discriminator Real Loss: {d_loss_real}")
        print(f"Discriminator Fake Loss: {d_loss_fake}")
        print(f"Generator Loss: {g_loss}")
        print("=================================")

        model.d_loss_fake.append(d_loss_fake)
        model.d_loss_real.append(d_loss_real)
        model.g_loss.append(g_loss)

        model.w_dist = []
        model.batch_sizes = []
        model.discriminator_losses_fake = []
        model.discriminator_losses_real = []
        model.generator_losses = []
        model.w_dist = []
