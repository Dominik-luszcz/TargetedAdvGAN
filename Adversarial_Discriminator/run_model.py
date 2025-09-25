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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    recall_score,
)
from scipy.stats import gaussian_kde

# import pytorch_forecasting as pf


# from path_generation import compute_bond_SDE, compute_stock_SDE
from adversarialDiscriminator import *

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)


class StockDataset(Dataset):
    def __init__(self, stock_folder, split_file, attack, sep="_"):
        super().__init__()

        split = np.load(split_file, allow_pickle=True)
        train, val, test = (
            split.item()["train"],
            split.item()["val"],
            split.item()["test"],
        )
        real_train, fake_train = [], []
        real_val, fake_val = [], []
        real_test, fake_test = [], []

        for entry in Path(stock_folder).iterdir():
            if entry.suffix == ".csv":
                df = pd.read_csv(entry)
                real_adjprc = np.expand_dims(df["adjprc"].to_numpy(), axis=0)
                fake_adjprc = np.expand_dims(df[attack].to_numpy(), axis=0)

                if entry.name.split(sep)[0] in train:
                    real_train.append(real_adjprc)
                    fake_train.append(fake_adjprc)

                elif entry.name.split(sep)[0] in val:
                    real_val.append(real_adjprc)
                    fake_val.append(fake_adjprc)

                else:
                    real_test.append(real_adjprc)
                    fake_test.append(fake_adjprc)

        self.real_train, self.fake_train = torch.from_numpy(
            np.concatenate(real_train)
        ), torch.from_numpy(np.concatenate(fake_train))
        self.real_val, self.fake_val = torch.from_numpy(
            np.concatenate(real_val)
        ), torch.from_numpy(np.concatenate(fake_val))
        self.real_test, self.fake_test = torch.from_numpy(
            np.concatenate(real_test)
        ), torch.from_numpy(np.concatenate(fake_test))

        self.real_labels = torch.ones((len(real_train), 1))
        self.fake_labels = torch.zeros((len(fake_train), 1))
        self.real_train = torch.concatenate([self.real_train, self.real_labels], dim=-1)
        self.fake_train = torch.concatenate([self.fake_train, self.fake_labels], dim=-1)
        self.training_set = torch.concatenate([self.real_train, self.fake_train], dim=0)

        self.real_labels = torch.ones((len(real_val), 1))
        self.fake_labels = torch.zeros((len(fake_val), 1))
        self.real_val = torch.concatenate([self.real_val, self.real_labels], dim=-1)
        self.fake_val = torch.concatenate([self.fake_val, self.fake_labels], dim=-1)
        self.validation_set = torch.concatenate([self.real_val, self.fake_val], dim=0)

        self.real_labels = torch.ones((len(real_test), 1))
        self.fake_labels = torch.zeros((len(fake_test), 1))
        self.real_test = torch.concatenate([self.real_test, self.real_labels], dim=-1)
        self.fake_test = torch.concatenate([self.fake_test, self.fake_labels], dim=-1)
        self.test_set = torch.concatenate([self.real_test, self.fake_test], dim=0)

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, index):
        return self.training_set[index]


def train_on_one_stocks(
    data_files,
    attack,
    split_file,
    hidden_dim,
    batch_size,
    num_epochs,
    output_path,
    sep="_",
):

    dataset = StockDataset(
        stock_folder=data_files, split_file=split_file, attack=attack, sep=sep
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )  # , num_workers=19)

    model = AdversarialDiscriminator(hidden_dim=hidden_dim)

    train_callback = ModelCheckpoint(
        monitor="f1",
        mode="max",
        save_top_k=1,
        filename=f"best-model",
        verbose=True,
        dirpath=output_path,
    )

    # Init the trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        accumulate_grad_batches=1,
        logger=False,
        callbacks=[
            train_callback,
            AdversarialDiscriminatorCallback(
                dataset.validation_set, dataset.training_set
            ),
        ],
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        max_epochs=num_epochs,  # max_steps=MAX_ITERATIONS,
        enable_progress_bar=True,
        max_time="00:20:00:00",
        default_root_dir=output_path,
    )

    trainer.fit(model, dataloader)

    # torch.save(model.state_dict(), f"{output_path}/adv_model_last_epoch.pth")

    # model.load_state_dict(torch.load('vGan_model.pth'))
    # test_datset = StockDataset(data_files, training_split_file=training_split_file, mode='test')

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_model = AdversarialDiscriminator.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), f"{output_path}/adv_gan.pt")

    plt.plot(model.train_loss, color="blue", label="Train Loss")
    plt.plot(model.val_loss, color="orange", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(f"{output_path}/train_curve.png")
    plt.close()

    test(model, dataset.test_set, output_path)

    # inference_test(model, noise_dim)

    return


def test(model: AdversarialDiscriminator, test_set, output_path):

    data, labels = test_set[:, :-1].unsqueeze(-1), test_set[:, -1].unsqueeze(-1)

    with torch.no_grad():
        discriminator_outputs = model.forward(data)

    labels_np = labels.detach().cpu().numpy()
    discriminator_outputs_np = discriminator_outputs.detach().cpu().numpy()

    pred = discriminator_outputs_np > model.threshold
    f1 = f1_score(labels_np, pred)
    acc = accuracy_score(labels_np, pred)
    precision = precision_score(labels_np, pred)
    recall = recall_score(labels_np, pred)
    specificity = (
        np.sum((np.array(labels_np) == np.array(pred)) & (np.array(labels_np) == 0))
        / np.sum(np.array(labels_np) == 0)
        if np.sum(np.array(labels_np) == 0) > 0
        else 0
    )
    kappa = cohen_kappa_score(labels_np, pred)
    mcc = matthews_corrcoef(labels_np, pred)

    matrix = confusion_matrix(labels_np, pred)
    matrix_p = confusion_matrix(labels_np, pred, normalize="true")
    plt.matshow(matrix_p, cmap="Blues")
    plt.xticks(range(len(matrix)), ["Fake", "Real"], fontsize=10)
    plt.yticks(range(len(matrix)), ["Fake", "Real"], fontsize=10)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    p = (np.min(matrix_p) + np.max(matrix_p)) / 2
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            color = "white" if matrix_p[i, j] > p else "black"
            value = f"{round(100 * matrix_p[i, j], 2)}%\n({matrix[i, j]})"
            plt.text(
                j,
                i,
                value,
                c=color,
                horizontalalignment="center",
                verticalalignment="center",
            )

    plt.savefig(f"{output_path}/test_cm.png", bbox_inches="tight")
    plt.close()

    m = [
        {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
            "MCC": mcc,
            "kappa": kappa,
        }
    ]

    df = pd.DataFrame(m)
    df.to_csv(f"{output_path}/test_metrics.csv")


if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Started job at {t1}")

    output_path = r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdversarialDiscriminatorOutputs\agan_results"
    initialize_directory(output_path)

    # train_on_one_stocks(data_files="Attack_Outputs/first300_relative_eps_0.02_all", attack="slope_up_adjprc", split_file='training_split.npy', hidden_dim=16, batch_size=32, #32 for subsample
    #       num_epochs=200, output_path=output_path )#load_model_path=r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdvGAN_A_5\mapping_gan.pt")

    train_on_one_stocks(
        data_files="AdvGAN_A_v4_2_5_results/sample_recordings",
        attack="agan_adjprc",
        split_file="training_split_agan.npy",
        hidden_dim=16,
        batch_size=32,  # 32 for subsample
        num_epochs=200,
        output_path=output_path,
        sep=".csv",
    )  # load_model_path=r"C:\Users\annal\TarAdvGAN_v3\TargetedAdvGAN\AdvGAN_A_5\mapping_gan.pt")

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
