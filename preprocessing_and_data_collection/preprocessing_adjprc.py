import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


from get_CRSP_data import *
from helper_functions import *

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

            standardScalar = RobustScaler()

            if df["adjprc"].isnull().any():
                df["adjprc"] = df["adjprc"].interpolate(method="linear")
                print(entry)

            if df["adjprc"].isnull().any():
                print(entry)

            # Take the log adjprc
            #df["adjprc"] = np.log(df["adjprc"])


            # Need to compute rolling means and stdev for the adjprc
            # For the NaN means, fill them in with the corresponding adjprc (only affects the first k, where k = window size)
            df["adjprc_rolling_mean3"] = df["adjprc"].rolling(3).mean()
            df["adjprc_rolling_mean3"] = df["adjprc_rolling_mean3"].fillna(df["adjprc"])
            df["adjprc_rolling_mean5"] = df["adjprc"].rolling(5).mean()
            df["adjprc_rolling_mean5"] = df["adjprc_rolling_mean5"].fillna(df["adjprc"])
            df["adjprc_rolling_mean10"] = df["adjprc"].rolling(10).mean()
            df["adjprc_rolling_mean10"] = df["adjprc_rolling_mean10"].fillna(df["adjprc"])
            df["adjprc_rolling_mean20"] = df["adjprc"].rolling(20).mean()
            df["adjprc_rolling_mean20"] = df["adjprc_rolling_mean20"].fillna(df["adjprc"])

            # For the NaN stdevs, fill them in with the 0
            df["adjprc_rolling_stdev3"] = df["adjprc"].rolling(3).std()
            df["adjprc_rolling_stdev3"] = df["adjprc_rolling_stdev3"].fillna(df["adjprc_rolling_stdev3"].mean())
            df["adjprc_rolling_stdev5"] = df["adjprc"].rolling(5).std()
            df["adjprc_rolling_stdev5"] = df["adjprc_rolling_stdev5"].fillna(df["adjprc_rolling_stdev5"].mean())
            df["adjprc_rolling_stdev10"] = df["adjprc"].rolling(10).std()
            df["adjprc_rolling_stdev10"] = df["adjprc_rolling_stdev10"].fillna(df["adjprc_rolling_stdev10"].mean())
            df["adjprc_rolling_stdev20"] = df["adjprc"].rolling(20).std()
            df["adjprc_rolling_stdev20"] = df["adjprc_rolling_stdev20"].fillna(df["adjprc_rolling_stdev20"].mean())

            # Compute the log returns
            df["log_returns"] = np.log(df["adjprc"]) - np.log(df["adjprc"].shift(1))
            df["log_returns"] = df["log_returns"].fillna(0)
            df["ROC_5"] = df["adjprc"].pct_change(5)
            df["ROC_5"] = df["ROC_5"].fillna(0)


            df["ema_10"] = df["adjprc"].ewm(span=10).mean()
            df["ema_20"] = df['adjprc'].ewm(span=20).mean()


            # add the specific day of the week as a feature, as stock 
            # markets tend to have a lull on Monday and improve as the week goes on
            df["date"] = pd.to_datetime(df["date"])
            df["adjprc_day"] = df["date"].dt.day_of_week

            df["time_idx"] = np.arange(len(df))

            # Save the feature file in the output_folder
            df = df.drop(columns=["permno", "date"])
            # Use StandardScalar to scale the features (adjprc is scaled when setting up the dataset)
            to_scale = ["adjprc_rolling_mean3", "adjprc_rolling_mean5", "adjprc_rolling_mean10", "adjprc_rolling_mean20",
                "adjprc_rolling_stdev3", "adjprc_rolling_stdev5", "adjprc_rolling_stdev10", "adjprc_rolling_stdev20",
                "log_returns", "ROC_5", "ema_10", "ema_20"]
            df[to_scale] = standardScalar.fit_transform(df[to_scale])
            df.to_csv(f"{output_path}/{df["ticker"].to_numpy()[0]}-features.csv", index=False)


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

    #preprocess("SP500_data", "SP500_Features_adjprc")
    determine_split("sp500_tickers.csv", 'SP500_Features_adjprc')
    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")