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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from lightning.pytorch.tuner import Tuner


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
                df["adjprc_day"] = df["adjprc_day"].astype(str)
                all_training_data.append(df)
    all_training_data = pd.concat(all_training_data, ignore_index=True)


    all_val_data = []
    for file in val_files:
        if os.path.exists(f"{feature_path}/{file}-features.csv"):
            df = pd.read_csv(f"{feature_path}/{file}-features.csv")
            if len(df) < min_length:
                continue
            else:
                df["adjprc_day"] = df["adjprc_day"].astype(str)
                all_val_data.append(df)
    all_val_data = pd.concat(all_val_data, ignore_index=True)
    
    all_testing_data = []
    for file in testing_files:
        if os.path.exists(f"{feature_path}/{file}-features.csv"):
            df = pd.read_csv(f"{feature_path}/{file}-features.csv")
            if len(df) < min_length:
                continue
            else:
                df["adjprc_day"] = df["adjprc_day"].astype(str)
                all_testing_data.append(df)
    all_testing_data = pd.concat(all_testing_data, ignore_index=True)

    # Set up the TimeSeriesDataSets
    train_datset = TimeSeriesDataSet(
        all_training_data,
        group_ids=["ticker"],
        target="adjprc",
        time_idx="time_idx",
        max_encoder_length=100,
        min_prediction_length=20,
        max_prediction_length=20,
        time_varying_known_categoricals=["adjprc_day"],
        time_varying_unknown_reals=["adjprc","rolling_mean5",
                                    "rolling_mean10","rolling_mean20",
                                    "rolling_stdev5","rolling_stdev10","rolling_stdev20",
                                    "log_returns", "roc5", "ema_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust"),
    )

    val_datset = TimeSeriesDataSet(
        all_val_data,
        group_ids=["ticker"],
        target="adjprc",
        time_idx="time_idx",
        max_encoder_length=100,
        min_prediction_length=20,
        max_prediction_length=20,
        time_varying_known_categoricals=["adjprc_day"],
        time_varying_unknown_reals=["adjprc","rolling_mean5",
                                    "rolling_mean10","rolling_mean20",
                                    "rolling_stdev5","rolling_stdev10","rolling_stdev20",
                                    "log_returns", "roc5", "ema_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust"),
    )

    test_datset = TimeSeriesDataSet(
        all_testing_data,
        group_ids=["ticker"],
        target="adjprc",
        time_idx="time_idx",
        max_encoder_length=100,
        min_prediction_length=20,
        max_prediction_length=20,
        time_varying_known_categoricals=["adjprc_day"],
        time_varying_unknown_reals=["adjprc","rolling_mean5",
                                    "rolling_mean10","rolling_mean20",
                                    "rolling_stdev5","rolling_stdev10","rolling_stdev20",
                                    "log_returns", "roc5", "ema_5", "ema_10", "ema_20"],
        target_normalizer=GroupNormalizer(groups=["ticker"], method="robust"),
    )

    return train_datset, val_datset, test_datset

'''
Initialize and train the model given the training sets and loaders
'''
def train(train_dataset: TimeSeriesDataSet, train_dataloader:DataLoader, val_dataloader:DataLoader):

    # Initialize the NHiTS Model
    model = NHiTS.from_dataset(train_dataset, weight_decay=1e-4,
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
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=15, verbose=False, mode="min")
    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        #limit_train_batches=200,
        enable_checkpointing=True,
        num_sanity_val_steps=0
    )

    lr = find_optimal_learning_rate(trainer, model, train_dataloader, val_dataloader)
    
    # Start training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load and save the best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), "./NHITS_forecasting_model.pt")
    params = best_model.hparams
    params["loss"] = QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    torch.save(params, "./NHITS_params.pt")


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

    total_mae = 0
    # Plot for each stock
    pdf_path = f"{mode}_latest_stock_predictions_adjprc.pdf"
    with PdfPages(pdf_path) as pdf:
        for stock, data in stock_projection.items():
            fig = plt.figure(figsize=(8, 6))
            mase = mean_absolute_error(data["true"], data["pred"])
            total_mae += mase
            plt.plot(data["true"], label="Actual", color="blue")
            plt.plot(data["pred"], label="Predicted", color="orange")
            plt.title(f"Predicted vs Actual for stock {stock} (last window), MAE: {mase}")
            plt.legend()
            plt.xlabel("Day")
            plt.ylabel("adjprc")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    total_mae = total_mae / len(stock_projection.items())
    print(f"TOTAL MAE: {total_mae}")

def test(model: NHiTS, test_dataloader:DataLoader, test_dataset: TimeSeriesDataSet, mode: str = "test", output_path:str = "."):
       # Make the predictions
    predictions = model.predict(test_dataloader, return_x=True, mode="prediction", trainer_kwargs=dict(accelerator="cpu"))

    # Get all our test tickers
    tickers = test_dataset.decoded_index["ticker"].unique()

    # Get the times, predictions and ground truth values
    time_idx = predictions.x["decoder_time_idx"]
    actuals = predictions.x["decoder_target"]
    preds = predictions.output

    # by def, each time_idx must start at 300 since it is the max encoder length
    starts = torch.argwhere(time_idx[:, 0] == 300)
    ends = torch.cat([(starts)[1:, :], torch.tensor([len(time_idx)]).unsqueeze(-1)], dim=0)
    starts_and_ends = torch.cat([starts, ends], dim=1)

    # For each ticker, find the average prediction and plot it
    pdf_path = f"{mode}_full_prediction2.pdf"
    with PdfPages(pdf_path) as pdf:
        ticker_index = 0
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        
        for start, end in starts_and_ends:
            corresponding_time = time_idx[start:end].flatten().numpy()
            corresponding_actuals = actuals[start:end].flatten().numpy()
            corresponding_pred = preds[start:end].flatten().numpy()

            df = pd.DataFrame({
                "time": corresponding_time,
                "Actual": corresponding_actuals,
                "Prediction": corresponding_pred
            })
            # Group by time and find the average
            df = df.groupby("time").agg({
                "Prediction" : 'mean',
                'Actual': 'mean'
            })
            # Calculate the errors
            mae = mean_absolute_error(df["Actual"], df["Prediction"])
            total_mae += mae

            rmse = root_mean_squared_error(df["Actual"], df["Prediction"])
            total_rmse += rmse
            mape = mean_absolute_percentage_error(df["Actual"], df["Prediction"])
            total_mape += mape
            fig = plt.figure(figsize=(14, 6))
            plt.plot(df.index, df["Actual"], label="Actual", color="blue")
            plt.plot(df.index, df["Prediction"], label="Predicted", color="orange")
            plt.title(f"Predicted vs Actual for stock {tickers[ticker_index]} (all), MAE: {mae}")
            plt.legend()
            plt.xlabel("Day")
            plt.ylabel("adjprc")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            df.to_csv(f"{output_path}/{tickers[ticker_index]}.csv")

            ticker_index += 1
    total_mae = total_mae / len(starts_and_ends)
    total_rmse = total_rmse / len(starts_and_ends)
    total_mape = total_mape / len(starts_and_ends)
    with open(f"{output_path}/avg_metrics2.txt", 'a') as f:
        f.write(f"Average MAE: {total_mae}\n")
        f.write(f"Average RMSE: {total_rmse}\n")
        f.write(f"Average MAPE: {total_mape}\n")

def find_optimal_learning_rate(trainer: pl.Trainer, model: NHiTS, train_dataloader, val_dataloader):

    result = Tuner(trainer).lr_find(
        model = model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-05,
        max_lr=1e-01
    )
    print(f"Suggested Learning Rate from the Tuner: {result.suggestion()}")
    model.hparams.learning_rate = result.suggestion()
    return result.suggestion()

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # Set up the datasets
    train_dataset, val_dataset, test_dataset = set_up_dataset("SP500_Features_adjprc", "training_split.npy")

    train_dataset.save("train_dataset.npy")

    # Set up the dataloaders
    train_loader = train_dataset.to_dataloader(train=True, batch_size=500, num_workers=19, persistent_workers=True)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=500, num_workers=19, persistent_workers=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=500, num_workers=19, persistent_workers=True)


    #Train the model
    train(train_dataset, train_loader, val_loader)

    #Load the best model and test
    model_state_dict = torch.load("NHITS_forecasting_model.pt")
    # model = NHiTS.from_dataset(train_dataset, weight_decay=1e-3,
    #                            hidden_size=64, log_val_interval=1, batch_normalization=True,
    #                            loss=QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))

    params = torch.load("./NHITS_params.pt", weights_only=False)
    params["loss"] = QuantileLoss(quantiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    model = NHiTS(**params)
    model.load_state_dict(model_state_dict)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=500, num_workers=19, persistent_workers=True)

    #test(model, train_loader, train_dataset, mode='train')
    #test_last_window(model, test_loader, test_dataset, mode='test')
    test(model, test_loader, test_dataset, mode='test', output_path="Forecast_outputs")


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

    # d = {
    #     'encoder_cat': torch.tensor([]),
    #     'encoder_cont': torch.zeros((2, 300, 13)), # batchsize X lookback X num_features
    #     'encoder_lengths': torch.tensor([300]),
    #     'decoder_cat': torch.tensor([]),
    #     'decoder_cont': torch.zeros((2, 50, 13)), # batchsize X forward projection X num_features
    #     'decoder_lengths': torch.tensor([50]),
    #     'encoder_target': torch.zeros((2, 300)),
    #     'decoder_target': torch.zeros((2, 50)),
    #     'target_scale': torch.zeros((1, 2)) # [center, scale] -> transform = (y-center)/scale -> inverse = y * scale + center
    # }

    # output = model(d)
    '''
    output[0] = tensor of shape [2, 50, 7] # 7 because we have 7 quantiles, so 2 batches, 50 forward proj, 7 quantiles, quantile 0.5 is the prediction so it would be index 3 for the prediction
    output[1] = tensor of shape [2, 300, 1] # 2 batches, 300 for the lookback
    output[2] = tuple of length 4: 
        output[2][0]: tensor of shape [2, 300, 1]
        output[2][1]: tensor of shape [2, 300, 1]
        output[2][2]: tensor of shape [2, 300, 1]
        output[2][3]: tensor of shape [2, 300, 1]
    output[3] = tuple of length 4:
        output[3][0]: tensor of shape [2, 50, 7] # these ones would be the scaling, to scale back to raw adjprc rather than the robust scaled adjprc
        output[3][1]: tensor of shape [2, 50, 7]
        output[3][2]: tensor of shape [2, 50, 7]
        output[3][3]: tensor of shape [2, 50, 7]

    '''


    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")