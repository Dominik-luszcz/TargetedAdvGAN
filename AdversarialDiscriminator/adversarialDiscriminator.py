import pandas as pd
import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from submodels import *
import random
import os
import torch.autograd as autograd
from sklearn.metrics import f1_score, precision_recall_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AdversarialDiscriminator(pl.LightningModule):
    def __init__(
        self, hidden_dim, input_dim=1, output_dim=1, lr=1e-04, weight_decay=1e-05
    ):
        super().__init__()

        self.discriminator = Discriminator(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loss = []
        self.val_loss = []
        self.threshold = 0
        self.best_metric = None
        self.double()
        self.save_hyperparameters()

    def forward(self, data):
        return self.discriminator(data)

    def training_step(self, batch):

        # Batch should be of shape B, seq_len, 2
        data, labels = batch[:, :-1].unsqueeze(-1), batch[:, -1].unsqueeze(-1)

        discriminator_outputs = self.forward(data)

        loss = nn.functional.binary_cross_entropy_with_logits(
            discriminator_outputs, labels
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return [optimizer]


class AdversarialDiscriminatorCallback(pl.Callback):
    def __init__(self, validation_set, training_set):
        super().__init__()
        self.validation_set = validation_set
        self.training_set = training_set

    def get_threshold(self, pred, labels):
        precision, recall, thresholds = precision_recall_curve(labels, pred)
        zero_precision, zero_recall = np.argwhere(precision == 0), np.argwhere(
            recall == 0
        )
        combined = np.intersect1d(zero_precision, zero_recall)
        precision, recall, thresholds = (
            np.delete(precision, combined),
            np.delete(recall, combined),
            np.delete(thresholds, combined),
        )

        f1 = 2 * (precision * recall) / (precision + recall)

        best_f1 = np.argwhere(f1 == f1.max())

        return thresholds[best_f1][0]

    def on_train_epoch_end(self, trainer, model: AdversarialDiscriminator):

        val_data, val_labels = self.validation_set[:, :-1].unsqueeze(
            -1
        ), self.validation_set[:, -1].unsqueeze(-1)
        train_data, train_labels = self.training_set[:, :-1].unsqueeze(
            -1
        ), self.training_set[:, -1].unsqueeze(-1)

        with torch.no_grad():
            discriminator_outputs_val = model.forward(val_data)
            discriminator_outputs_train = model.forward(train_data)

        val_labels_np = val_labels.detach().cpu().numpy()
        discriminator_outputs_val_np = discriminator_outputs_val.detach().cpu().numpy()
        threshold = self.get_threshold(discriminator_outputs_val_np, val_labels_np)

        pred = discriminator_outputs_val_np > threshold
        f1 = f1_score(val_labels.detach().cpu().numpy(), pred)

        val_loss = nn.functional.binary_cross_entropy_with_logits(
            discriminator_outputs_val, val_labels
        )
        model.val_loss.append(val_loss)
        train_loss = nn.functional.binary_cross_entropy_with_logits(
            discriminator_outputs_train, train_labels
        )
        model.train_loss.append(train_loss)

        if model.best_metric == None:
            model.best_metric = f1
            model.threshold = threshold
        else:
            if f1 > model.best_metric:
                model.best_metric = f1
                model.threshold = threshold

        model.log("f1", f1)
        print(f"F1 Score: {f1}")
