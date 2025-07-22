import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
#torch.set_num_threads(40)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path
import random
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
import pytorch_forecasting as pf


#from path_generation import compute_bond_SDE, compute_stock_SDE
from adv_network import *

import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)

def get_days(recording: pd.DataFrame):
        recording["date"] = pd.to_datetime(recording["date"])
        recording["adjprc_day"] = recording["date"].dt.day_of_week
        days = torch.from_numpy(recording["adjprc_day"].values) # we do not touch categorical features since it would be obvious
        days = days.type(torch.int)
        return days


class SingleStockDataset(Dataset):
    def __init__(self, stock_folder, ticker, num_samples, sample_size):
        super().__init__()
 

        self.max_return = None
        self.min_return = None
        self.num_samples = num_samples
        self.sample_size = sample_size

        try:
            df = pd.read_csv(f"{stock_folder}/{ticker}.csv")
            self.training_stock = df['adjprc'].to_numpy()
            self.days = get_days(df).double()
        except:
            raise ValueError(f"The ticker {ticker} is not in the provided stock folder {stock_folder}")

        self.log_returns = torch.from_numpy(np.log(self.training_stock[1:] / self.training_stock[:-1]))
        self.training_stock = torch.from_numpy(self.training_stock)

        self.unscaled_returns = self.log_returns


        #scale the training data between -1 and 1
        self.max_return = torch.quantile(self.log_returns, 1)
        self.min_return = torch.quantile(self.log_returns, 0)

        print(f"Max Return: {self.max_return}")
        print(f"Min Return: {self.min_return}")

        self.log_returns = 2 * ((self.log_returns - self.min_return) / (self.max_return - self.min_return)) - 1

        self.time_series_metrics = self.get_metrics()
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        start = random.randint(1, len(self.log_returns) - self.sample_size - 20)
        return self.training_stock[start-1], self.days[start-1: start + self.sample_size], self.log_returns[start : start + self.sample_size], self.log_returns[start + self.sample_size: start + self.sample_size + 20]
    
    def get_metrics(self):
        mean = torch.mean(self.unscaled_returns)
        std = torch.std(self.unscaled_returns)

        z = (self.unscaled_returns - mean) / std
        skew = torch.mean(z ** 3)
        kurtosis = torch.mean(z ** 4) - 3

        return {
            'mean': mean,
            'stdev': std,
            'skew': skew,
            'kurtosis': kurtosis
        }


def train_on_one_stocks(data_files, ticker, num_samples, sample_size, batch_size, num_epochs, output_path, model, load_model_path: None):

    dataset = SingleStockDataset(stock_folder=data_files, ticker=ticker, num_samples=num_samples, sample_size=sample_size)

    dataloader = DataLoader(dataset, batch_size=batch_size)#, num_workers=19)

    model = AdversarialNetwork(sample_size=sample_size,
                               model = model, alpha=1, scale_max=dataset.max_return, scale_min=dataset.min_return,
                               plot_paths=output_path)
    
    if load_model_path != None:
        model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))
    
    train_callback = ModelCheckpoint(
            monitor="loss",
            mode="min",
            save_top_k=1,
            filename=f"best-model",
            verbose=True,
            dirpath=output_path
    )

    model.epoch_betas = [20, 50, 80, 110, 130, 140]
    model.beta = 0.5
    model.beta_scale = 1.25
    model.alpha = 1
    model.c = 5 
    model.d = 2
    # Init the trainer
    trainer = pl.Trainer(devices='auto', accelerator='auto', accumulate_grad_batches=1, logger=False, callbacks=[train_callback, DCGAN_Callback(dataset.unscaled_returns, dataset.log_returns, dataset.days, dataset.training_stock, num_to_sample=64)], 
                         num_sanity_val_steps=0, enable_checkpointing=True, max_epochs=num_epochs,#max_steps=MAX_ITERATIONS,
                        enable_progress_bar=True, max_time='00:10:00:00', default_root_dir=output_path)
    
    trainer.fit(model, dataloader)

    plot_loss_functions(model, output_path)

    torch.save(model.state_dict(), f"{output_path}/adv_model.pth")


    #model.load_state_dict(torch.load('vGan_model.pth'))
    #test_datset = StockDataset(data_files, training_split_file=training_split_file, mode='test')

    # best_model_path = trainer.checkpoint_callback.best_model_path
    # print(best_model_path)
    # best_model = VanillaGAN.load_from_checkpoint(best_model_path)
    # torch.save(best_model.state_dict(), f"dc_gan.pt")

    #test_gan(model, noise_dim)

    return

def plot_loss_functions(model: AdversarialNetwork, output_path):
        # plot example fake data
    plt.plot(model.d_loss_real, color='blue', label='Discriminator Loss (Real)')
    plt.plot(model.d_loss_fake, color='orange', label='Discriminator Loss (Fake)')
    plt.plot(model.g_loss, color='red', label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for D and G')
    plt.legend()
    plt.savefig(f'{output_path}/g_d_loss_curve.png')
    plt.close()


    plt.plot(model.final_w_dists, color='blue', label='W Dist Approximations')
    plt.xlabel('Epoch')
    plt.ylabel('W Dist')
    plt.title('W Dist Approximations')
    plt.legend()
    plt.savefig(f'{output_path}/w_dist.png')
    plt.close()

    plt.plot(model.g_pens, color='blue', label='Gradient Penalties')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Penalty Magnitude')
    plt.title('Gradient Penalty over time')
    plt.legend()
    plt.savefig(f'{output_path}/g_pen.png')
    plt.close()

    # plot example fake data
    plt.plot(model.adversarial_losses, color='blue', label='Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for Adversarial Loss')
    plt.legend()
    plt.savefig(f'{output_path}/adv_loss_curve.png')
    plt.close()

    plt.plot(model.final_losses, color='blue', label='Final Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Final Loss Curve')
    plt.legend()
    plt.savefig(f'{output_path}/final_loss_curve.png')
    plt.close()
   



def maximum_mean_discrepency(X, Y, gamma=1):

    xx = rbf_kernel(X, X, gamma)
    yy = rbf_kernel(Y, Y, gamma)
    xy = rbf_kernel(X, Y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def sample_stats(model: AdversarialNetwork, log_returns, num_to_sample, output_dir):

        # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
        real_means = []
        real_stdevs = []
        real_iqr = []
        real_skew = []
        real_kurtosis = []
        real_samples = []
        for i in range(num_to_sample):
            start = random.randint(0, len(log_returns) - model.sample_size)
            interval = log_returns[start : start + model.sample_size]
            real_samples.append(interval.unsqueeze(0).detach().cpu().numpy())

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

        real_samples = np.concat(real_samples)
        np.save("real_samples.npy", real_samples)
        fake_output = fake_output.squeeze(-1).detach().cpu().numpy()
        mmd = maximum_mean_discrepency(real_samples, fake_output, gamma=1)

        with open(f"{output_dir}/sample_stats_wgan2.txt", 'a') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Real Mean: {real_means}, Fake Mean: {fake_means}\n")
            file.write(f"Real Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
            file.write(f"Real iqr: {real_iqr}, Fake iqr: {fake_iqr}\n")
            file.write(f"Real skew: {real_skew}, Fake skew: {fake_skew}\n")
            file.write(f"Real kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n")
            file.write(f"Loss: {loss}\n")
            file.write(f"MMD: {mmd}\n")
            file.write("=" * 50 + "\n")

        real_kde = gaussian_kde(real_samples.flatten())
        fake_kde = gaussian_kde(fake_output.flatten())

        grid = np.linspace(min(real_samples.min(), fake_output.min()), max(real_samples.max(), fake_output.max()), 1000)

        real_pdf = real_kde(grid)
        fake_pdf = fake_kde(grid)
        #fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        plt.plot(grid, real_pdf, label='Real', color="red")
        plt.plot(grid, fake_pdf, label='Synthetic', color="blue")
        plt.legend()
        plt.xlabel('Log_return')
        plt.ylabel('Density')
        plt.savefig(f'{output_dir}/wgan_kde2.png')
        plt.close()
    
        plt.hist(real_samples.flatten(), label='Real', color="red", alpha=0.6, bins=100)
        plt.hist(fake_output.flatten(), label='Synthetic', color="blue", alpha=0.6, bins=100)
        plt.legend()
        plt.xlabel('Log_return')
        plt.ylabel('Density')
        plt.savefig(f'{output_dir}/wgan_hist2.png')
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fo = fake_output[0]
        ro = real_samples[0]
        ax[0].plot(fo)
        ax[0].set_xlabel('Time (days)')
        ax[0].set_ylabel('Log Returns')
        ax[0].set_title('Example GAN Log Return')

        ax[1].plot(ro)
        ax[1].set_xlabel('Time (days)')
        ax[1].set_title('Example Real Log Return')
        plt.savefig(f'{output_dir}/example_wgan_return.png')
        plt.close()

        

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # model = DCGAN.load_from_checkpoint(r"C:\Users\annal\WGan_A_50_simpler\best-model.ckpt")
    # dataset = SingleStockDataset(stock_folder="SP500_Filtered", ticker='A', num_samples=500, sample_size=50)
    
    # sample_stats(model, log_returns=dataset.unscaled_returns, num_to_sample=2000, output_dir='.')

    # output_path = '/scratch/a/alim/dominik/WGan_A_250epochs'
    # output_path = '/scratch/a/alim/dominik/WGan_A_50_simpler'
    # output_path = '/scratch/a/alim/dominik/WGan_A_50_simpler_350'
    # initialize_directory(output_path) 

    # train_on_one_stocks(data_files="/home/a/alim/dominik/SP500_Filtered", ticker='A', num_samples=384, sample_size=350, batch_size=32, #32 for subsample
    #       num_epochs=250, output_path=output_path, noise_dim=32,
    #       generator_hidden_dim=64, generator_output_dim=1, discriminator_hidden_dim=64)

    # output_path = '/scratch/a/alim/dominik/AdvGAN_A'
    # initialize_directory(output_path) 

    # model_state_dict = torch.load("NHITS_forecasting_model.pt")
    # params = torch.load("./NHITS_params.pt", weights_only=False)
    # params["loss"] = pf.QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    # model = pf.NHiTS(**params)
    # model.load_state_dict(model_state_dict)
    # model.eval()


    # train_on_one_stocks(data_files="/home/a/alim/dominik/SP500_Filtered", ticker='A', num_samples=384, sample_size=99, batch_size=32, #32 for subsample
    #       num_epochs=500, output_path=output_path, model=model)

    output_path = './AdvGAN_A_15'
    initialize_directory(output_path) 

    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = pf.QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    model = pf.NHiTS(**params)
    model.load_state_dict(model_state_dict)
    model.eval()


    train_on_one_stocks(data_files="SP500_Filtered", ticker='A', num_samples=384, sample_size=99, batch_size=32, #32 for subsample
          num_epochs=150, output_path=output_path, model=model, load_model_path=r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdvGAN_A_5\mapping_gan.pt")

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")