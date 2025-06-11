import pandas as pd
import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl
import torch.nn as nn
from sub_models import *
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance_nd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DCGAN(pl.LightningModule):
    def __init__(self, generator_hidden_dim, noise_dim, generator_output_dim, discriminator_hidden_dim, scale_max = None, scale_min = None, lr=0.0001, plot_paths='.'):
        super().__init__()

        self.lr = lr

        self.generator_hidden_dim = generator_hidden_dim
        self.noise_dim = noise_dim
        self.output_dim = generator_output_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.plot_paths = plot_paths
        self.scale_max = scale_max
        self.scale_min = scale_min

        self.generator = Generator(output_dim=generator_output_dim)
        self.discriminator = Discriminator(input_dim=1, hidden_dim=discriminator_hidden_dim, output_dim=1)

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

        self.save_hyperparameters()


    def training_step(self, batch):
        # get the optimizers for the generator and discriminator
        generator_optimizer, discriminator_optimizer = self.optimizers()

        # batch would give me batches of real data
        real_data = batch
        real_data = real_data.unsqueeze(-1).to(DEVICE)
        b, seq_len, dim = real_data.shape
        real_data_labels = torch.ones((b, 1), dtype=torch.float64, device=DEVICE) - 0.15
        fake_data_labels = torch.zeros((b, 1), dtype=torch.float64, device=DEVICE) + 0.15

        # Step 1: we need to train the discriminator
        discriminator_optimizer.zero_grad()

        # First we need to train on the real data
        discriminator_output_real = self.discriminator(real_data)
        # because we classify per ticker/recording we take the average to get size b x 1
        discriminator_output_real = discriminator_output_real.mean(dim=1)
        discriminator_loss_real = nn.functional.binary_cross_entropy(discriminator_output_real, real_data_labels)
        discriminator_loss_real.backward()

        # Next we need to train the discriminator on the fake data
        z = torch.randn(b, self.noise_dim, dtype=torch.float64, device=DEVICE).unsqueeze(-1)
        fake_data = self.generator(z)
        discriminator_output_fake = self.discriminator(fake_data.detach())
        discriminator_output_fake = discriminator_output_fake.mean(dim=1)
        discriminator_loss_fake = nn.functional.binary_cross_entropy(discriminator_output_fake, fake_data_labels)
        discriminator_loss_fake.backward()

        self.discriminator_losses_real.append( discriminator_loss_real.detach())
        self.discriminator_losses_fake.append( discriminator_loss_fake.detach())

        discriminator_optimizer.step()

        # Step 2: Train the generator
        generator_optimizer.zero_grad()
        fake_data = self.generator(z)
        discriminator_output_fake = self.discriminator(fake_data)
        discriminator_output_fake = discriminator_output_fake.mean(dim=1)
        # Now we want to update the generator gradients to make our fake data be classified as real data
        generator_loss = nn.functional.binary_cross_entropy(discriminator_output_fake, real_data_labels + 0.15) 
        generator_loss.backward()

        # generator_grad = 0
        # for param in self.generator.parameters():
        #     if param.grad is not None:
        #         generator_grad += param.grad.abs().sum().item()
        
        # print(generator_grad)



        # Maybe try a sort of stdev loss
        real_stdev = real_data.std()
        fake_stdev = fake_data.std()
        #stdev_loss = fake_stdev / (real_stdev + 1e-06) + real_stdev / (real_stdev + 1e-06)
        #stdev_loss.backward()
        generator_optimizer.step()

        self.generator_losses.append(generator_loss.detach())
        self.batch_sizes.append(b)

        wass_dist = wasserstein_distance_nd(real_data.squeeze(-1).detach().numpy(), fake_data.squeeze(-1).detach().numpy())
        self.w_dist.append(wass_dist)
        #self.log("wass_dist", wass_dist, on_step=False, on_epoch=True, batch_size=b)
    
    def forward(self, x):
        return self.generator(x)
    

    def on_train_epoch_end(self):

        generator_losses = torch.Tensor(self.generator_losses)
        discriminator_losses_real = torch.Tensor(self.discriminator_losses_real)
        discriminator_losses_fake = torch.Tensor(self.discriminator_losses_fake)
        wass_dists = torch.Tensor(self.w_dist)
        batch_sizes = torch.Tensor(self.batch_sizes)

        d_loss_real = sum(discriminator_losses_real *  batch_sizes) / sum(batch_sizes)
        d_loss_fake = sum(discriminator_losses_fake *  batch_sizes) / sum(batch_sizes)
        w_dist = sum(wass_dists *  batch_sizes) / sum(batch_sizes)
        g_loss = sum(generator_losses *  batch_sizes) / sum(batch_sizes)

        self.log("wass_dist", w_dist)
        

        self.eval()
        with torch.no_grad():
            z = torch.randn(1, self.noise_dim, dtype=torch.float64).unsqueeze(-1)
            fake_output = self.generator(z)
            self.example_outputs.append(fake_output)

        self.train()
        # plot example fake data
        # first we need to scale the fake data to match that of the log returns
        fake_output = fake_output.squeeze(-1).squeeze(0).detach()
        if self.scale_max != None and self.scale_min != None:
            fake_output = (self.scale_max - self.scale_min) * (0.5 * (fake_output + 1)) + self.scale_min
        plt.plot(fake_output.numpy())
        plt.xlabel('Time (days)')
        plt.ylabel('Log Returns')
        plt.title('Example log return created from GAN')
        plt.savefig(f'{self.plot_paths}/example_gan_output_epoch{self.current_epoch}.png')
        plt.close()


        print('=================================')
        print(f"Discriminator Real Loss: {d_loss_real}")
        print(f"Discriminator Fake Loss: {d_loss_fake}")
        print(f"Generator Loss: {g_loss}")
        print(f"Wasserstein Distance: {w_dist}")
        print('=================================')

        self.d_loss_fake.append(d_loss_fake)
        self.d_loss_real.append(d_loss_real)
        self.g_loss.append(g_loss)

        self.w_dist = []
        self.batch_sizes = []
        self.discriminator_losses_fake = []
        self.discriminator_losses_real = []
        self.generator_losses = []


    
    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-03)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-05)

        return generator_optimizer, discriminator_optimizer