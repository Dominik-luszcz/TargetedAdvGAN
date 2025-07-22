import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import torch

from get_CRSP_data import *
from helper_functions import *

def exponential_moving_average(adjprc: torch.Tensor, span: int):

    alpha = 2.0 / (span + 1.0)

    ema = torch.zeros_like(adjprc) 
    ema[0] = adjprc[0]

    # formula taken from the pandas definition
    for i in range(1, len(adjprc)):
        ema[i] = (1 - alpha) * adjprc[i-1] + alpha * adjprc[i]

    return ema

def feature_generation(adjprc: torch.Tensor):

    weight5 = torch.ones((1,1,5))/5
    weight10 = torch.ones((1,1,10))/10
    weight20 = torch.ones((1,1,20))/20
    rolling_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight5, stride=1)

    rolling_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight10, stride=1)

    rolling_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0), weight20, stride=1)

    # stdevs: Variance = E[x^2] - E[x]^2
    # so E[x] = rolling_meanX => E[x]^2 = rolling_meanX ^ 2
    rollingExp_mean5 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight5, stride=1)
    rollingExp_mean10 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight10, stride=1)
    rollingExp_mean20 = torch.nn.functional.conv1d(adjprc.unsqueeze(0).unsqueeze(0) ** 2, weight20, stride=1)

    rolling_stdev5 = torch.sqrt(rollingExp_mean5 - rolling_mean5**2 + 1e-06)
    rolling_stdev10 = torch.sqrt(rollingExp_mean10 - rolling_mean10**2 + 1e-06)
    rolling_stdev20 = torch.sqrt(rollingExp_mean20 - rolling_mean20**2 + 1e-06)
    # but we have padding and the model fills the first x values with something (either adjprc or mean rolling_stdev) so we have to fix that
    rolling_mean5 = rolling_mean5.squeeze(0).squeeze(0)
    rolling_mean5 = torch.cat([adjprc[:4], rolling_mean5])
    rolling_stdev5 = rolling_stdev5.squeeze(0).squeeze(0)
    rolling_stdev5 = torch.cat([torch.full(size=[4], fill_value=torch.mean(rolling_stdev5.squeeze(0).squeeze(0)).item()), rolling_stdev5])

    rolling_mean10 = rolling_mean10.squeeze(0).squeeze(0)
    rolling_mean10 = torch.cat([adjprc[:9], rolling_mean10])
    rolling_stdev10 = rolling_stdev10.squeeze(0).squeeze(0)
    rolling_stdev10 = torch.cat([torch.full(size=[9], fill_value=torch.mean(rolling_stdev10.squeeze(0).squeeze(0)).item()), rolling_stdev10])

    rolling_mean20 = rolling_mean20.squeeze(0).squeeze(0)
    rolling_mean20 = torch.cat([adjprc[:19], rolling_mean20])
    rolling_stdev20 = rolling_stdev20.squeeze(0).squeeze(0)
    rolling_stdev20 = torch.cat([torch.full(size=[19], fill_value=torch.mean(rolling_stdev20.squeeze(0).squeeze(0)).item()), rolling_stdev20])

    # log returns 
    log_returns = torch.log(adjprc[1:]/adjprc[:-1]) 
    log_returns = torch.cat([torch.tensor([0.0]), log_returns])

    # ROC (percetn change of 5)
    roc5 = (adjprc[5:] - adjprc[:-5]) / adjprc[:-5]
    roc5 = torch.cat([torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]), roc5])

    # exponential moving averages
    ema_5 = exponential_moving_average(adjprc=adjprc, span=5)
    ema_10 = exponential_moving_average(adjprc=adjprc, span=10)
    ema_20 = exponential_moving_average(adjprc=adjprc, span=20)

    features = torch.stack([rolling_mean5, rolling_mean10, rolling_mean20, 
                             rolling_stdev5, rolling_stdev10, rolling_stdev20, 
                             log_returns, roc5, ema_5, ema_10, ema_20], dim=1)


    # Because we are doing 1 recording at a time, the standard scalar for teh features (not target) are just the mean and stdev of the recording
    # Similarly, the quantiles are the same for the target for the robust scalar for adjprc

    avg = torch.mean(features, dim=0)
    stdev = torch.std(features, dim=0)

    features = (features - avg) / stdev

    # median = torch.median(adjprc, dim=0).values
    # q75 = torch.quantile(adjprc, q=0.75, dim=0)
    # q25 = torch.quantile(adjprc, q=0.25, dim=0)
    # #adjprc = (adjprc - median) / (q75 - q25)

    # scale_ = q75 - q25
    # center_ = median

    center_ = np.median(adjprc.detach().numpy())
    scale_ = (np.quantile(adjprc.detach().numpy(), q=0.75) - np.quantile(adjprc.detach().numpy(), q=0.25)) / 2

    features = torch.hstack([adjprc.unsqueeze(-1), features])

    return features, scale_, center_
        

'''
Given a data_path of csv files with crsp data, preprocess the data and save it to output_path
'''
def preprocess(data_path: str, output_path: str):
    initialize_directory(output_path)

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            # If the stock has less than 600 days of data then ignore it (should be moved to collect_data or when we make the split)
            if len(df) < 1250:
                continue

            adjprc = torch.from_numpy(df["adjprc"].values)

            # If missing values just skip
            if torch.isnan(adjprc).any():
                continue


            features, center_, scale_ = feature_generation(adjprc)

            columns = ["adjprc", "rolling_mean5", "rolling_mean10", "rolling_mean20", 
                             "rolling_stdev5", "rolling_stdev10", "rolling_stdev20", 
                             "log_returns", "roc5", "ema_5", "ema_10", "ema_20"]
            
            features_df = pd.DataFrame(features.detach().cpu().numpy(), columns=columns)
            
            # add the specific day of the week as a feature, as stock 
            # markets tend to have a lull on Monday and improve as the week goes on
            df["date"] = pd.to_datetime(df["date"])
            features_df["adjprc_day"] = df["date"].dt.day_of_week


            features_df["time_idx"] = np.arange(len(df))

            features_df["ticker"] = df["ticker"]

            features_df.to_csv(f"{output_path}/{df["ticker"].to_numpy()[0]}-features.csv", index=False)


def determine_split(ticker_path: str, feature_path: str) -> None:
    tickers = pd.read_csv(ticker_path)

    to_include = []

    for entry in Path(feature_path).iterdir():
        if entry.suffix == ".csv":
            ticker = entry.name.split("-features")[0]
            if ticker in tickers["Symbol"].to_numpy():
                to_include.append(ticker)

    filtered_data = tickers[tickers["Symbol"].isin(to_include)]

    filtered_data['label'] = pd.qcut(filtered_data["Stock Price"], q=8, labels=False)

    # Use a stratied split based on the stock price
    train_stocks, test_val_stocks = train_test_split(
        filtered_data,
        train_size=0.75,
        stratify=filtered_data["label"],
        random_state=23
    )

    test_stocks, val_stocks = train_test_split(
        test_val_stocks,
        train_size=0.6,
        stratify=test_val_stocks["label"],
        random_state=23
    )

    print(f"Train size: {len(train_stocks)}, Test Size: {len(test_stocks)}, Val Size: {len(val_stocks)}")

    # Save the training splits in a npy file
    np.save("training_split.npy", {"train": train_stocks["Symbol"].to_numpy(),
                                     "test": test_stocks["Symbol"].to_numpy(),
                                     "val": val_stocks["Symbol"].to_numpy()})



if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    preprocess("SP500_data", "SP500_Features_adjprc")
    determine_split("sp500_tickers.csv", 'SP500_Features_adjprc')
    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")