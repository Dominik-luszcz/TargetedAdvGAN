import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# torch.set_num_threads(40)
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


# from path_generation import compute_bond_SDE, compute_stock_SDE
from adv_network import *

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    days = torch.from_numpy(
        recording["adjprc_day"].values
    )  # we do not touch categorical features since it would be obvious
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
            self.training_stock = df["adjprc"].to_numpy()
            self.days = get_days(df).double()
        except:
            raise ValueError(
                f"The ticker {ticker} is not in the provided stock folder {stock_folder}"
            )

        self.log_returns = torch.from_numpy(
            np.log(self.training_stock[1:] / self.training_stock[:-1])
        )
        self.training_stock = torch.from_numpy(self.training_stock)

        self.unscaled_returns = self.log_returns

        # scale the training data between -1 and 1
        self.max_return = torch.quantile(self.log_returns, 1)
        self.min_return = torch.quantile(self.log_returns, 0)

        print(f"Max Return: {self.max_return}")
        print(f"Min Return: {self.min_return}")

        self.log_returns = (
            2
            * (
                (self.log_returns - self.min_return)
                / (self.max_return - self.min_return)
            )
            - 1
        )

        self.time_series_metrics = self.get_metrics()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start = random.randint(1, len(self.log_returns) - self.sample_size - 20)
        return (
            self.training_stock[start - 1],
            self.days[start - 1 : start + self.sample_size],
            self.log_returns[start : start + self.sample_size],
            self.log_returns[start + self.sample_size : start + self.sample_size + 20],
        )

    def get_metrics(self):
        mean = torch.mean(self.unscaled_returns)
        std = torch.std(self.unscaled_returns)

        z = (self.unscaled_returns - mean) / std
        skew = torch.mean(z**3)
        kurtosis = torch.mean(z**4) - 3

        return {"mean": mean, "stdev": std, "skew": skew, "kurtosis": kurtosis}


def train_on_one_stocks(
    data_files,
    ticker,
    num_samples,
    sample_size,
    batch_size,
    num_epochs,
    output_path,
    nhits_model,
    load_model_path=None,
):

    dataset = SingleStockDataset(
        stock_folder=data_files,
        ticker=ticker,
        num_samples=num_samples,
        sample_size=sample_size,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)  # , num_workers=19)

    model = AdversarialNetwork(
        sample_size=sample_size,
        model=nhits_model,
        alpha=1,
        scale_max=dataset.max_return,
        scale_min=dataset.min_return,
        plot_paths=output_path,
    )

    if load_model_path != None:
        best_model = AdversarialNetwork.load_from_checkpoint(
            load_model_path, strict=False, model=nhits_model, sample_size=sample_size
        )

    train_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=1,
        filename=f"best-model",
        verbose=True,
        dirpath=output_path,
    )

    model.epoch_betas = [60]
    model.beta = 0.4  # 0.25, 0.35, 0.35, 0.4
    model.beta_scale = 1.25
    model.alpha = 1
    model.c = 5
    model.d = 2
    # Init the trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        accumulate_grad_batches=1,
        logger=False,
        callbacks=[
            train_callback,
            DCGAN_Callback(
                dataset.unscaled_returns,
                dataset.log_returns,
                dataset.days,
                dataset.training_stock,
                num_to_sample=64,
            ),
        ],
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        max_epochs=num_epochs,  # max_steps=MAX_ITERATIONS,
        enable_progress_bar=True,
        max_time="00:10:00:00",
        default_root_dir=output_path,
    )

    trainer.fit(model, dataloader)

    torch.save(model.state_dict(), f"{output_path}/adv_model.pth")

    # model.load_state_dict(torch.load('vGan_model.pth'))
    # test_datset = StockDataset(data_files, training_split_file=training_split_file, mode='test')

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_model = AdversarialNetwork.load_from_checkpoint(
        best_model_path, strict=False, model=nhits_model, sample_size=sample_size
    )
    torch.save(best_model.state_dict(), f"{output_path}/adversarial_network.pt")

    plot_loss_functions(model, output_path)

    # inference_test(model, noise_dim)

    return


def plot_loss_functions(model: AdversarialNetwork, output_path):
    # plot example fake data
    plt.plot(model.d_loss_real, color="blue", label="Discriminator Loss (Real)")
    plt.plot(model.d_loss_fake, color="orange", label="Discriminator Loss (Fake)")
    plt.plot(model.g_loss, color="red", label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve for D and G")
    plt.legend()
    plt.savefig(f"{output_path}/g_d_loss_curve.png")
    plt.close()

    plt.plot(model.final_w_dists, color="blue", label="W Dist Approximations")
    plt.xlabel("Epoch")
    plt.ylabel("W Dist")
    plt.title("W Dist Approximations")
    plt.legend()
    plt.savefig(f"{output_path}/w_dist.png")
    plt.close()

    plt.plot(model.g_pens, color="blue", label="Gradient Penalties")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Penalty Magnitude")
    plt.title("Gradient Penalty over time")
    plt.legend()
    plt.savefig(f"{output_path}/g_pen.png")
    plt.close()

    adversarial_losses = [l.detach().cpu().numpy() for l in model.adversarial_losses]
    # plot example fake data
    plt.plot(adversarial_losses, color="blue", label="Adversarial Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Adversarial Loss")
    plt.legend()
    plt.savefig(f"{output_path}/adv_loss_curve.png")
    plt.close()

    final_losses = [l.detach().cpu().numpy() for l in model.final_losses]
    plt.plot(final_losses, color="blue", label="Final Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final Loss Curve")
    plt.legend()
    plt.savefig(f"{output_path}/final_loss_curve.png")
    plt.close()


def maximum_mean_discrepency(X, Y, gamma=1):

    xx = rbf_kernel(X, X, gamma)
    yy = rbf_kernel(Y, Y, gamma)
    xy = rbf_kernel(X, Y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def sample_stats(
    model: AdversarialNetwork, dataset: SingleStockDataset, num_to_sample, output_dir
):

    random.seed(33)
    # 1. Sample n intervals of 400 days and compute stats like kurtosis and skew
    num_to_sample = num_to_sample
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
        start = random.randint(
            1, len(dataset.unscaled_returns) - model.sample_size - 20
        )
        interval = dataset.unscaled_returns[start : start + model.sample_size]
        scaled_interval = dataset.log_returns[start : start + model.sample_size]
        f = dataset.unscaled_returns[
            start + model.sample_size : start + model.sample_size + 20
        ].to(DEVICE)
        forecasts[i] = f

        d = dataset.days[start - 1 : start + model.sample_size]
        price = dataset.training_stock[start - 1]
        condition = scaled_interval.to(DEVICE)
        conditions[i] = condition
        intervals.append(interval)

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
        real_pred_scaled = real_pred  # (model.scale_max - model.scale_min) * ((real_pred + 1)/2) + model.scale_min
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

    recording_paths = f"{output_dir}/sample_recordings"
    if not os.path.exists(recording_paths):
        os.mkdir(recording_paths)
    for i in range(real_adjprc.shape[0]):
        df = pd.DataFrame(
            data=np.concatenate(
                [
                    real_adjprc[i].unsqueeze(-1).detach().cpu().numpy(),
                    adv_adjprc[i].unsqueeze(-1).detach().cpu().numpy(),
                ],
                axis=-1,
            ),
            columns=["adjprc", "agan_adjprc"],
        )

        df.to_csv(f"{recording_paths}/sample_{i}.csv")

    # 2. Run model in white box setting
    real_outputs, time_idx = model.call_model(real_adjprc, days)
    # real_predictions = self.get_predictions(real_outputs, time_idx) # if we just predict once we dont need to scatter_bin and get avg

    fake_outputs, _ = model.call_model(adv_adjprc, days)
    # fake_predictions = self.get_predictions(fake_outputs, time_idx)

    # 3. Compute the adversarial loss
    # slope = (fake_outputs[:, -1] - fake_outputs[:, 0]) / 20
    direction = model.target_direction * -1
    # adversarial_loss = torch.mean(-1 * model.c * torch.exp(direction * model.d * slope))
    # #a_loss = torch.nn.functional.l1_loss(fake_outputs, real_pred_adjprc)

    # # Compute the final loss and return
    # a_loss = torch.mean(model.beta * adversarial_loss)

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
    # Compute the final loss and return
    # f_loss = s_loss + a_loss

    temp = [i.unsqueeze(0) for i in intervals]
    real_samples = np.concat(temp, axis=0)

    y_mean_real = torch.mean(real_outputs, dim=1).unsqueeze(-1)

    numerator_real = ((x - x_mean) * (real_outputs - y_mean_real)).sum(dim=1)
    denom_real = ((x - x_mean) ** 2).sum(dim=1)

    slope_real = numerator_real / denom_real

    g_slope_real = real_outputs[:, -1] / real_outputs[:, 0]
    g_slope_fake = fake_outputs[:, -1] / fake_outputs[:, 0]

    def get_gamma(real_samples, fake_samples):
        all_data = torch.concat(
            [torch.from_numpy(real_samples), torch.from_numpy(fake_samples)], axis=0
        )
        differences = torch.pdist(all_data)[:1000] ** 2
        gamma = 1.0 / 2 * torch.median(differences)
        return gamma.detach().cpu().numpy()

    def normalize_adjprc(real_adjprc, adv_adjprc):
        return (real_adjprc) / np.mean(real_adjprc), (adv_adjprc) / np.mean(adv_adjprc)

    mmd = maximum_mean_discrepency(
        real_samples,
        fake_output.squeeze(-1).detach().cpu().numpy(),
        gamma=get_gamma(real_samples, fake_output.squeeze(-1).detach().cpu().numpy()),
    )

    adjprc_normed = normalize_adjprc(
        real_adjprc.detach().cpu().numpy(), adv_adjprc.detach().cpu().numpy()
    )

    mmd_adjprc = maximum_mean_discrepency(
        adjprc_normed[0],
        adjprc_normed[1],
        gamma=get_gamma(adjprc_normed[0], adjprc_normed[1]),
    )

    with open(f"{output_dir}/sample_stats_wgan2.txt", "a") as file:
        file.write("=" * 50 + "\n")
        file.write(f"Real Mean: {real_means}, Fake Mean: {fake_means}\n")
        file.write(f"Real Stdev: {real_stdevs}, Fake Stdev: {fake_stdevs}\n")
        file.write(f"Real iqr: {real_iqr}, Fake iqr: {fake_iqr}\n")
        file.write(f"Real skew: {real_skew}, Fake skew: {fake_skew}\n")
        file.write(f"Real kurtosis: {real_kurtosis}, Fake kurtosis: {fake_kurtosis}\n")
        file.write(f"Loss (similarity): {s_loss}\n")
        file.write(f"MMD (log return): {mmd}\n")
        file.write(f"MMD (adjprc): {mmd_adjprc}\n")
        file.write(f"Average LS Slope (real): {slope_real.mean()}\n")
        file.write(f"Average LS Slope (fake): {slope.mean()}\n")
        file.write(f"Average G Slope (real): {g_slope_real.mean()}\n")
        file.write(f"Average G Slope (fake): {g_slope_fake.mean()}\n")
        file.write("=" * 50 + "\n")

    real_kde = gaussian_kde(real_samples.flatten())
    fake_kde = gaussian_kde(fake_output.flatten())

    grid = np.linspace(
        min(real_samples.min(), fake_output.min()),
        max(real_samples.max(), fake_output.max()),
        1000,
    )

    real_pdf = real_kde(grid)
    fake_pdf = fake_kde(grid)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.plot(grid, real_pdf, label="Real", color="red")
    plt.plot(grid, fake_pdf, label="Synthetic", color="blue")
    plt.legend()
    plt.xlabel("Log_return")
    plt.ylabel("Density")
    plt.savefig(f"{output_dir}/wgan_kde2.png")
    plt.close()

    plt.hist(real_samples.flatten(), label="Real", color="red", alpha=0.6, bins=100)
    plt.hist(
        fake_output.flatten(), label="Synthetic", color="blue", alpha=0.6, bins=100
    )
    plt.legend()
    plt.xlabel("Log_return")
    plt.ylabel("Density")
    plt.savefig(f"{output_dir}/wgan_hist2.png")
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.tight_layout()
    plt.subplots_adjust(
        wspace=0.25, hspace=0.25, left=0.1, right=0.9, top=0.9, bottom=0.1
    )
    fo = fake_output[3]
    ro = real_samples[3]
    fp = fake_outputs[3].detach().cpu().numpy()
    rp = real_outputs[3].detach().cpu().numpy()
    r = real_pred_adjprc[3].detach().cpu().numpy()
    fa = adv_adjprc[3].detach().cpu().numpy()
    ra = real_adjprc[3].detach().cpu().numpy()
    ax[0, 0].plot(fo)
    ax[0, 0].set_xlabel("Time (days)")
    ax[0, 0].set_ylabel("Log Returns")
    ax[0, 0].set_title("Example GAN Log Return")

    ax[0, 1].plot(ro)
    ax[0, 1].set_xlabel("Time (days)")
    ax[0, 1].set_title("Example Real Log Return")

    ax[1, 0].plot(fa, label="Adversarial Adjprc")
    ax[1, 0].plot(ra, label="Real Adjprc")
    ax[1, 0].set_xlabel("Time (days)")
    ax[1, 0].set_ylabel("Adjprc")
    ax[1, 0].legend()
    ax[1, 0].set_title("Real vs Adversarial Adjprc")

    ax[1, 1].plot(fp, label="Adversarial Prediction")
    ax[1, 1].plot(r, label="Ground Truth")
    ax[1, 1].plot(rp, label="Real Prediction")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("Time (days)")
    ax[1, 1].set_title("Prediction Forecasts")
    plt.savefig(f"{output_dir}/example_agan_return.png")
    plt.close()

    for i in range(2, 11, 1):
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plt.tight_layout()
        plt.subplots_adjust(
            wspace=0.25, hspace=0.25, left=0.1, right=0.9, top=0.9, bottom=0.1
        )
        fp = fake_outputs[i * 3].detach().cpu().numpy()
        rp = real_outputs[i * 3].detach().cpu().numpy()
        r = real_pred_adjprc[i * 3].detach().cpu().numpy()
        fa = adv_adjprc[i * 3].detach().cpu().numpy()
        ra = real_adjprc[i * 3].detach().cpu().numpy()

        ax[0].plot(fa, label="Adversarial Adjprc")
        ax[0].plot(ra, label="Real Adjprc")
        ax[0].set_xlabel("Time (days)")
        ax[0].set_ylabel("Adjprc")
        ax[0].legend()
        ax[0].set_title("Real vs Adversarial Adjprc")

        ax[1].plot(fp, label="Adversarial Prediction")
        ax[1].plot(r, label="Ground Truth")
        ax[1].plot(rp, label="Real Prediction")
        ax[1].legend()
        ax[1].set_xlabel("Time (days)")
        ax[1].set_title("Prediction Forecasts")
        plt.savefig(f"{output_dir}/example_agan_prediction_{i * 3}.png")
        plt.close()


def agan_sample_split(num_to_sample):
    from sklearn.model_selection import train_test_split

    indicies = np.arange(0, num_to_sample)

    train, test = train_test_split(indicies, test_size=0.15)

    train, val = train_test_split(train, test_size=0.13333333333333333)

    train_files = [f"sample_{i}" for i in train]
    val_files = [f"sample_{i}" for i in val]
    test_files = [f"sample_{i}" for i in test]

    np.save(
        "training_split_agan.npy",
        {"train": train_files, "test": test_files, "val": val_files},
    )


if __name__ == "__main__":
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

    output_path = "./AdvGAN_A_v4_2_5_results"
    # initialize_directory(output_path)

    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = pf.QuantileLoss(
        quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    )
    model = pf.NHiTS(**params)
    model.load_state_dict(model_state_dict)
    model.eval()

    dataset = SingleStockDataset(
        stock_folder="SP500_Filtered", ticker="A", num_samples=512, sample_size=99
    )

    # dataloader = DataLoader(dataset, batch_size=batch_size)#, num_workers=19)

    adv_net = AdversarialNetwork(
        sample_size=99,
        model=model,
        alpha=1,
        scale_max=dataset.max_return,
        scale_min=dataset.min_return,
        plot_paths=output_path,
    )
    load_model_path = (
        r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdvGAN_A_v4_2_5\best-model.ckpt"
    )

    best_model = AdversarialNetwork.load_from_checkpoint(
        load_model_path,
        strict=False,
        model=model,
        sample_size=99,
        scale_max=dataset.max_return,
        scale_min=dataset.min_return,
    )
    sample_stats(best_model, dataset, num_to_sample=2000, output_dir=output_path)

    # train_on_one_stocks(data_files="SP500_Filtered", ticker='A', num_samples=512, sample_size=99, batch_size=32, #32 for subsample
    #       num_epochs=50, output_path=output_path, nhits_model=model, load_model_path=r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdvGAN_A_v4_2_5\best-model.ckpt")

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
