import pandas as pd
import numpy as np
from datetime import datetime
from pytorch_forecasting import NHiTS, TimeSeriesDataSet, SMAPE, Baseline, QuantileLoss, MAPE
import os
from torch.utils.data import DataLoader
import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.data import GroupNormalizer
import matplotlib.pyplot as plt
from torch import nn
from matplotlib.backends.backend_pdf import PdfPages


'''
Set up the dataset by converting the csv files into one large TimeSeriesDataSet
'''
def set_up_dataset(feature_path:str, split_file: str):

    split = np.load(split_file, allow_pickle=True)

    training_files = split.item()["train"]
    testing_files = split.item()["test"]
    val_files = split.item()["val"]

    # If we have less than 300 days of data do not consider it (DONE IN PREPROCESSING, SHOULD BE SAFE TO REMOVE)
    min_length = 300
    all_training_data = []
    for file in training_files:
        if os.path.exists(f"{feature_path}/{file}-features.csv"):
            df = pd.read_csv(f"{feature_path}/{file}-features.csv")
            if len(df) < min_length:
                continue
            else:
                all_training_data.append(df)
    all_training_data = pd.concat(all_training_data, ignore_index=True)


    all_val_data = []
    for file in val_files:
        if os.path.exists(f"{feature_path}/{file}-features.csv"):
            df = pd.read_csv(f"{feature_path}/{file}-features.csv")
            if len(df) < min_length:
                continue
            else:
                all_val_data.append(df)
    all_val_data = pd.concat(all_val_data, ignore_index=True)
    
    all_testing_data = []
    for file in testing_files:
        if os.path.exists(f"{feature_path}/{file}-features.csv"):
            df = pd.read_csv(f"{feature_path}/{file}-features.csv")
            if len(df) < min_length:
                continue
            else:
                all_testing_data.append(df)
    all_testing_data = pd.concat(all_testing_data, ignore_index=True)

    # Set up the TimeSeriesDataSets
    train_datset = TimeSeriesDataSet(
        all_training_data,
        group_ids=["ticker"],
        target="returns",
        time_idx="time_idx",
        max_encoder_length=300,
        min_prediction_length=50,
        max_prediction_length=50,
        time_varying_known_reals=["time_idx", "returns_day"],
        time_varying_unknown_reals=["returns","returns_rolling_mean5",
                                    "returns_rolling_mean10","returns_rolling_mean20",
                                    "returns_rolling_stdev5","returns_rolling_stdev10","returns_rolling_stdev20"],
                                   # "log_returns", "ROC_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust")
    )

    val_datset = TimeSeriesDataSet(
        all_val_data,
        group_ids=["ticker"],
        target="returns",
        time_idx="time_idx",
        max_encoder_length=300,
        min_prediction_length=50,
        max_prediction_length=50,
        time_varying_known_reals=["time_idx", "returns_day"],
        time_varying_unknown_reals=["returns","returns_rolling_mean5",
                                    "returns_rolling_mean10","returns_rolling_mean20",
                                    "returns_rolling_stdev5","returns_rolling_stdev10","returns_rolling_stdev20"],
                                    #"log_returns", "ROC_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust")
    )

    test_datset = TimeSeriesDataSet(
        all_testing_data,
        group_ids=["ticker"],
        target="returns",
        time_idx="time_idx",
        max_encoder_length=300,
        min_prediction_length=50,
        max_prediction_length=50,
        time_varying_known_reals=["time_idx", "returns_day"],
        time_varying_unknown_reals=["returns","returns_rolling_mean5",
                                    "returns_rolling_mean10","returns_rolling_mean20",
                                    "returns_rolling_stdev5","returns_rolling_stdev10","returns_rolling_stdev20"],
                                    #"log_returns", "ROC_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust")
    )

    return train_datset, val_datset, test_datset

'''
Initialize and train the model given the training sets and loaders
'''
def train(train_dataset: TimeSeriesDataSet, train_dataloader:DataLoader, val_dataloader:DataLoader):

    # Initialize the NHiTS Model
    model = NHiTS.from_dataset(train_dataset, learning_rate = 1e-03, weight_decay=1e-4,
                               hidden_size=64, log_val_interval=1, batch_normalization=True,
                               loss=QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))

    # Initialize the checkpoint callback (to save on best validation)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
        verbose=True
    )
    
    # Initialize early stopping callback (same as the pytorch-forecasting docs)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=15, verbose=False, mode="min")
    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        #limit_train_batches=200,
        enable_checkpointing=True,
    )
    
    # Start training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load and save the best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), "./NHITS_forecasting_model_ret.pt")


'''
Test the model by predicting on the last possible window for the given test set.
'''
def test_last_window(model: NHiTS, test_dataloader:DataLoader, test_dataset: TimeSeriesDataSet, mode: str = "test"):
    # Make the predictions
    predictions = model.predict(test_dataloader, return_x=True, mode="prediction", trainer_kwargs=dict(accelerator="cpu"))

    # Get all our test tickers
    tickers = test_dataset.decoded_index["ticker"].unique()

    stock_projection = {}

    # Calculate the last index for each stock
    groups = predictions.x["groups"][:, 0]
    # Since the groups are sorted we can calculate the diffs to get the locations of the last index for each stock
    diffs = torch.diff(groups)
    # Since the groups are sorted the only places where we have a non-zero value is when we have a change in stock
    times_of_change = torch.nonzero(diffs)[:, 0]
    # Append the final index for the final stock
    lastest_index = torch.cat([times_of_change, torch.tensor([len(groups) - 1])])

    # Go through the last indexes (last prediction windows for each stock in the test set)
    # and append the required data to plot the time series
    for ticker_index, index in enumerate(lastest_index):
        tick = tickers[ticker_index]
        time_idx = predictions.x["decoder_time_idx"][index]
        t_proj = predictions.x["decoder_target"][index].detach().cpu().numpy()
        p_proj = predictions.output[index].numpy()
        stock_projection[tick] = {"time" : time_idx, "true": t_proj, "pred": p_proj}

    # Plot for each stock
    pdf_path = f"{mode}_latest_stock_predictions_ret.pdf"
    with PdfPages(pdf_path) as pdf:
        for stock, data in stock_projection.items():
            fig = plt.figure(figsize=(8, 6))
            plt.plot(data["true"], label="Actual", color="blue")
            plt.plot(data["pred"], label="Predicted", color="orange")
            plt.title(f"Predicted vs Actual for stock {stock} (last window)")
            plt.legend()
            plt.xlabel("Day")
            plt.ylabel("returns (multiplied by 100)")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def test(model: NHiTS, test_dataloader:DataLoader, test_dataset: TimeSeriesDataSet, mode: str = "test"):
    # Make the predictions
    predictions = model.predict(test_dataloader, return_x=True, mode="prediction", trainer_kwargs=dict(accelerator="cpu"))

    # Get all our test tickers
    tickers = test_dataset.decoded_index["ticker"].unique()

    time_idx = predictions.x["decoder_time_idx"]
    actuals = predictions.x["decoder_target"]
    preds = predictions.output

    # by def, each time_idx must start at 300 since it is the max encoder length
    starts = torch.argwhere(time_idx[:, 0] == 300)

    ends = torch.cat([(starts)[1:, :], torch.tensor([len(time_idx)]).unsqueeze(-1)], dim=0)

    starts_and_ends = torch.cat([starts, ends], dim=1)


    pdf_path = f"{mode}_full_prediction_ret.pdf"
    with PdfPages(pdf_path) as pdf:
        ticker_index = 0
        for start, end in starts_and_ends:
            corresponding_time = time_idx[start:end].flatten().numpy()
            corresponding_actuals = actuals[start:end].flatten().numpy()
            corresponding_pred = preds[start:end].flatten().numpy()

            df = pd.DataFrame({
                "time": corresponding_time,
                "Actual": corresponding_actuals,
                "Prediction": corresponding_pred
            })

            df = df.groupby("time").agg({
                "Prediction" : 'mean',
                'Actual': 'mean'
            })

            fig = plt.figure(figsize=(14, 6))
            plt.plot(df.index, df["Actual"], label="Actual", color="blue")
            plt.plot(df.index, df["Prediction"], label="Predicted", color="orange")
            plt.title(f"Predicted vs Actual for stock {tickers[ticker_index]} (all)")
            plt.legend()
            plt.xlabel("Day")
            plt.ylabel("returns")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            ticker_index += 1






    # # Go through the last indexes (last prediction windows for each stock in the test set)
    # # and append the required data to plot the time series
    # for ticker_index, index in enumerate(lastest_index):
    #     tick = tickers[ticker_index]
    #     time_idx = predictions.x["decoder_time_idx"][index]
    #     t_proj = predictions.x["decoder_target"][index].detach().cpu().numpy()
    #     p_proj = predictions.output[index].numpy()
    #     stock_projection[tick] = {"time" : time_idx, "true": t_proj, "pred": p_proj}

    # # Plot for each stock
    # pdf_path = f"{mode}_latest_stock_predictions.pdf"
    # with PdfPages(pdf_path) as pdf:
    #     for stock, data in stock_projection.items():
    #         fig = plt.figure(figsize=(8, 6))
    #         plt.plot(data["true"], label="Actual", color="blue")
    #         plt.plot(data["pred"], label="Predicted", color="orange")
    #         plt.title(f"Predicted vs Actual for stock {stock} (last window)")
    #         plt.legend()
    #         plt.xlabel("Day")
    #         plt.ylabel("returns (multiplied by 100)")
    #         pdf.savefig(fig, bbox_inches='tight')
    #         plt.close()
    

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # Set up the datasets
    train_dataset, val_dataset, test_dataset = set_up_dataset("SP500_Features_sampled_ret", "training_split.npy")

    # Set up the dataloaders
    train_loader = train_dataset.to_dataloader(train=True, batch_size=500, num_workers=19, persistent_workers=True)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=500, num_workers=19, persistent_workers=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=500, num_workers=19, persistent_workers=True)

    # Train the model
    train(train_dataset, train_loader, val_loader)

    # Load the best model and test
    model_state_dict = torch.load("NHITS_forecasting_model_ret.pt")
    model = NHiTS.from_dataset(train_dataset, learning_rate = 1e-03, weight_decay=1e-4,
                               hidden_size=64, log_val_interval=1, batch_normalization=True,
                               loss=QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
    model.load_state_dict(model_state_dict)
    train_loader = train_dataset.to_dataloader(train=False, batch_size=350, num_workers=19, persistent_workers=True)
    #test(model, train_loader, train_dataset, mode='train')
    test(model, test_loader, test_dataset, mode='test')

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")