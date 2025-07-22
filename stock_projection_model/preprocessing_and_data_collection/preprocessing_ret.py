import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
            if len(df) < 600:
                continue

            standardScalar = RobustScaler()

            # Some have null/nan but only for the first day (you cant calculate returns if you dont have the prior date)
            # Mainly because I am filtering the data for just the past decade
            df = df.tail(len(df) - 1)

            if df["returns"].isnull().any():
                df["returns"] = df["returns"].interpolate(method="linear")
                print(entry)

            if df["returns"].isnull().any():
                print(entry)

            # because of small values we need to scale the returns
            df["returns"] = df["returns"] * 100

            # Need to compute rolling means and stdev for the returns
            # For the NaN means, fill them in with the corresponding returns (only affects the first k, where k = window size)
            df["returns_rolling_mean3"] = df["returns"].rolling(3).mean()
            df["returns_rolling_mean3"] = df["returns_rolling_mean3"].fillna(df["returns"])
            df["returns_rolling_mean5"] = df["returns"].rolling(5).mean()
            df["returns_rolling_mean5"] = df["returns_rolling_mean5"].fillna(df["returns"])
            df["returns_rolling_mean10"] = df["returns"].rolling(10).mean()
            df["returns_rolling_mean10"] = df["returns_rolling_mean10"].fillna(df["returns"])
            df["returns_rolling_mean20"] = df["returns"].rolling(20).mean()
            df["returns_rolling_mean20"] = df["returns_rolling_mean20"].fillna(df["returns"])

            # For the NaN stdevs, fill them in with the 0
            df["returns_rolling_stdev3"] = df["returns"].rolling(3).std()
            df["returns_rolling_stdev3"] = df["returns_rolling_stdev3"].fillna(df["returns_rolling_stdev3"].mean())
            df["returns_rolling_stdev5"] = df["returns"].rolling(5).std()
            df["returns_rolling_stdev5"] = df["returns_rolling_stdev5"].fillna(df["returns_rolling_stdev5"].mean())
            df["returns_rolling_stdev10"] = df["returns"].rolling(10).std()
            df["returns_rolling_stdev10"] = df["returns_rolling_stdev10"].fillna(df["returns_rolling_stdev10"].mean())
            df["returns_rolling_stdev20"] = df["returns"].rolling(20).std()
            df["returns_rolling_stdev20"] = df["returns_rolling_stdev20"].fillna(df["returns_rolling_stdev20"].mean())

            # add the specific day of the week as a feature, as stock 
            # markets tend to have a lull on Monday and improve as the week goes on
            df["date"] = pd.to_datetime(df["date"])
            df["returns_day"] = df["date"].dt.day_of_week

            df["time_idx"] = np.arange(len(df))

            # Save the feature file in the output_folder
            df = df.drop(columns=["permno", "date"])
            # Use StandardScalar to scale the features (returns is scaled when setting up the dataset)
            to_scale = ["returns_rolling_mean3", "returns_rolling_mean5", "returns_rolling_mean10", "returns_rolling_mean20",
                "returns_rolling_stdev3", "returns_rolling_stdev5", "returns_rolling_stdev10", "returns_rolling_stdev20"]
            df[to_scale] = standardScalar.fit_transform(df[to_scale])
            df.to_csv(f"{output_path}/{df["ticker"].to_numpy()[0]}-features.csv", index=False)




if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    preprocess("SP500_data_sampled_ret", "SP500_Features_sampled_ret")
    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")