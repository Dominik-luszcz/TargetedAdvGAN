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


class DCGAN(pl.LightningModule):
    def __init__(self, generator_hidden_dim, noise_dim, generator_output_dim, discriminator_hidden_dim,
                 sample_size, n_critic = 5, lmda = 1, beta1 = 0.0, beta2=0.9,
                 scale_max = None, scale_min = None, lr=0.0001, plot_paths='.'):
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
        self.n_critic = n_critic
        self.lmda = lmda
        self.beta1 = beta1
        self.beta2 = beta2

        self.generator = Generator(output_dim=generator_output_dim, sample_size=sample_size)
        self.discriminator = Discriminator(input_dim=1, hidden_dim=discriminator_hidden_dim, output_dim=1)

        self.automatic_optimization = False

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

        self.save_hyperparameters()

    def compute_gradient_penalty(self, real_data: Tensor, fake_data: Tensor):

        # random interpolation factor alpha
        alpha = torch.rand(real_data.shape[0], 1, 1, device=DEVICE)
        alpha = alpha.expand_as(real_data)



        # calculate the gradients from the interpolation output
        with torch.autograd.set_detect_anomaly(True):
            # calculate the interpolation
            interpolation = alpha * real_data + (1-alpha) * fake_data
            interpolation = interpolation.requires_grad_(True)

            # feed the interpolation into the discriminator
            interpolation_output = self.discriminator(interpolation)
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
        # get the optimizers for the generator and discriminator
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # batch would give me batches of real data
        real_data = batch
        real_data = real_data.unsqueeze(-1).to(DEVICE)
        b, seq_len, dim = real_data.shape

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
            # First we need to train on the real data
            discriminator_output_real = self.discriminator(real_data)
            # because we classify per ticker/recording we take the average to get size b x 1
            discriminator_output_real = discriminator_output_real.mean()
            
            
    
            # Next we need to train the discriminator on the fake data
            z = torch.randn(b, real_data.shape[1], dtype=torch.float64, device=DEVICE).unsqueeze(-1)
            fake_data = self.generator(z)
            discriminator_output_fake = self.discriminator(fake_data.detach())
            discriminator_output_fake = discriminator_output_fake.mean()

            # Now we have to do the gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_data=real_data, fake_data=fake_data)
            # compute the loss
            # goal is to maximize E[D(real)] - E[D(fake)], so instead we minimize E[D(fake)] - E[D(real)]
            discriminator_loss = discriminator_output_fake - discriminator_output_real + gradient_penalty
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
        fake_data = self.generator(z)
        generated_discrim_output = self.discriminator(fake_data)
        generator_loss = -1 * generated_discrim_output.mean()
        generator_loss.backward()


        generator_optimizer.step()

        self.generator_losses.append(generator_loss.detach())
        self.batch_sizes.append(b)

    
    def forward(self, x):
        return self.generator(x)

    
    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-04, betas=(self.beta1, self.beta2))

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
        num_to_sample = 32
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
                plt.xlabel('Time (days)')
                plt.ylabel('Log Returns')
                plt.title(f'Example real Return: {model.sample_size} Days ')
                plt.savefig(f'{model.plot_paths}/Epoch_{model.current_epoch}/example_real_output_epoch{model.current_epoch}.png')
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
            z = torch.randn(num_to_sample, 50, dtype=torch.float64, device=DEVICE).unsqueeze(-1)
            fake_output = model.generator(z)

            # Need to scale back to log returns
            if model.scale_max != None and model.scale_min != None:

                fake_output = (model.scale_max - model.scale_min) * ((fake_output + 1)/2) + model.scale_min

            model.example_outputs.append(fake_output)

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

        loss = ((real_means - fake_means) ** 2 + (real_stdevs - fake_stdevs) ** 2 + 
                (real_skew - fake_skew) ** 2 + (real_kurtosis - fake_kurtosis) ** 2)


        with open(f"{model.plot_paths}/Epoch_{model.current_epoch}/generated_stats_epoch_{model.current_epoch}.txt", 'a') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Real Mean: {real_means}, Fake Mean: {fake_means}\n")
            file.write(f"Real Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Real iqr: {real_iqr}, Fake iqr: {fake_iqr}\n")
            file.write(f"Real skew: {real_skew}, Fake skew: {fake_skew}\n")
            file.write(f"Real kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n")
            file.write(f"Loss {loss}\n")
            file.write("=" * 50 + "\n")

        model.log('loss', loss)
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
        print(f"Discriminator Real Loss: {d_loss_real}")
        print(f"Discriminator Fake Loss: {d_loss_fake}")
        print(f"Generator Loss: {g_loss}")
        print(f"W Dist Approx: {w_dist}")
        print('=================================')

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