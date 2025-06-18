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

#from path_generation import compute_bond_SDE, compute_stock_SDE
from wgan_dc import *

import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)


class SingleStockDataset(Dataset):
    def __init__(self, stock_folder, ticker, num_samples, sample_size):
        super().__init__()
 

        self.max_return = None
        self.min_return = None
        self.num_samples = num_samples
        self.sample_size = sample_size

        try:
            self.training_stock = pd.read_csv(f"{stock_folder}/{ticker}.csv")['adjprc'].to_numpy()
        except:
            raise ValueError(f"The ticker {ticker} is not in the provided stock folder {stock_folder}")

        self.log_returns = torch.from_numpy(np.log(self.training_stock[1:] / self.training_stock[:-1]))


        self.unscaled_returns = self.log_returns




        #scale the training data between -1 and 1
        self.max_return = torch.quantile(self.log_returns, 1)
        self.min_return = torch.quantile(self.log_returns, 0)

        print(f"Max Return: {self.max_return}")
        print(f"Min Return: {self.min_return}")

        

        self.log_returns = 2 * ((self.log_returns - self.min_return) / (self.max_return - self.min_return)) - 1

        
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        start = random.randint(0, len(self.log_returns) - self.sample_size)
        return self.log_returns[start : start + self.sample_size]


def train_on_one_stocks(data_files, ticker, num_samples, sample_size, batch_size, num_epochs, output_path, noise_dim,
           generator_hidden_dim, generator_output_dim, discriminator_hidden_dim, lr=0.0001):

    dataset = SingleStockDataset(stock_folder=data_files, ticker=ticker, num_samples=num_samples, sample_size=sample_size)

    dataloader = DataLoader(dataset, batch_size=batch_size)#, num_workers=19)

    model = DCGAN(noise_dim=noise_dim, generator_hidden_dim=generator_hidden_dim, generator_output_dim=generator_output_dim, 
                       discriminator_hidden_dim=discriminator_hidden_dim, lr=lr, plot_paths=output_path,
                       scale_max = dataset.max_return, scale_min = dataset.min_return, sample_size=sample_size)
    
    train_callback = ModelCheckpoint(
            monitor="loss",
            mode="min",
            save_top_k=1,
            filename=f"best-model",
            verbose=True,
            dirpath=output_path
    )
    
    # Init the trainer
    trainer = pl.Trainer(devices='auto', accelerator='auto', accumulate_grad_batches=1, logger=False, callbacks=[train_callback, DCGAN_Callback(dataset.unscaled_returns)], 
                         num_sanity_val_steps=0, enable_checkpointing=True, max_epochs=num_epochs,#max_steps=MAX_ITERATIONS,
                        enable_progress_bar=True, max_time='00:03:00:00', default_root_dir=output_path)
    
    trainer.fit(model, dataloader)

    plot_loss_functions(model, output_path)

    torch.save(model.state_dict(), f"{output_path}/dc_gan_model.pth")


    #model.load_state_dict(torch.load('vGan_model.pth'))
    #test_datset = StockDataset(data_files, training_split_file=training_split_file, mode='test')

    # best_model_path = trainer.checkpoint_callback.best_model_path
    # print(best_model_path)
    # best_model = VanillaGAN.load_from_checkpoint(best_model_path)
    # torch.save(best_model.state_dict(), f"dc_gan.pt")

    #test_gan(model, noise_dim)

    return

def plot_loss_functions(model: DCGAN, output_path):
    # plot example fake data
    plt.plot(model.d_loss_real, color='blue', label='Discriminator Loss (Real)')
    plt.plot(model.d_loss_fake, color='orange', label='Discriminator Loss (Fake)')
    plt.plot(model.g_loss, color='red', label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for D and G')
    plt.legend()
    plt.savefig(f'{output_path}/loss_curve.png')
    plt.close()

    plt.plot(model.final_w_dists, color='blue', label='W Dist Approximations')
    plt.xlabel('Epoch')
    plt.ylabel('W Dist')
    plt.title('W Dist Approximations')
    plt.legend()
    plt.savefig(f'{output_path}/w_dist.png')
    plt.close()


def test_gan(model: DCGAN, noise_dim):

    with torch.no_grad():
        z = torch.randn(1, noise_dim, dtype=torch.float64).unsqueeze(-1)
        fake_data = model.generator(z)

    # plot example fake data
    plt.plot(fake_data.squeeze(-1).squeeze(0).detach().numpy())
    plt.xlabel('Time (days)')
    plt.ylabel('Log Returns')
    plt.title('Example log return created from GAN')
    plt.savefig('./example_gan_output.png')
    plt.close()

    




if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    output_path = '/scratch/a/alim/dominik/WGan_A_500epochs2'
    initialize_directory(output_path) 

    train_on_one_stocks(data_files="/home/a/alim/dominik/SP500_Filtered", ticker='A', num_samples=500, sample_size=400, batch_size=32, #32 for subsample
          num_epochs=500, output_path=output_path, noise_dim=32,
          generator_hidden_dim=64, generator_output_dim=1, discriminator_hidden_dim=64)

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")