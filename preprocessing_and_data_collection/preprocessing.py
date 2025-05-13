import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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

            standardScalar = StandardScaler()

            # Some have null/nan adjusted prices, so fill them with forward fill (ffill)
            if df["adjprc"].isnull().any():
                print(entry)
                df["adjprc"] = df["adjprc"].ffill()

            if df["adjprc"].isnull().any():
                print(entry)
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
            df["adjprc_rolling_stdev3"] = df["adjprc_rolling_stdev3"].fillna(0)
            df["adjprc_rolling_stdev5"] = df["adjprc"].rolling(5).std()
            df["adjprc_rolling_stdev5"] = df["adjprc_rolling_stdev5"].fillna(0)
            df["adjprc_rolling_stdev10"] = df["adjprc"].rolling(10).std()
            df["adjprc_rolling_stdev10"] = df["adjprc_rolling_stdev10"].fillna(0)
            df["adjprc_rolling_stdev20"] = df["adjprc"].rolling(20).std()
            df["adjprc_rolling_stdev20"] = df["adjprc_rolling_stdev20"].fillna(0)

            # add the specific day of the week as a feature, as stock 
            # markets tend to have a lull on Monday and improve as the week goes on
            df["date"] = pd.to_datetime(df["date"])
            df["adjprc_day"] = df["date"].dt.day_of_week

            df["time_idx"] = np.arange(len(df))

            # Save the feature file in the output_folder
            df = df.drop(columns=["permno", "date"])
            # Use StandardScalar to scale the features (adjprc is scaled when setting up the dataset)
            to_scale = ["adjprc_rolling_mean3", "adjprc_rolling_mean5", "adjprc_rolling_mean10", "adjprc_rolling_mean20",
                "adjprc_rolling_stdev3", "adjprc_rolling_stdev5", "adjprc_rolling_stdev10", "adjprc_rolling_stdev20"]
            df[to_scale] = standardScalar.fit_transform(df[to_scale])
            df.to_csv(f"{output_path}/{df["ticker"][0]}-features.csv", index=False)




if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    preprocess("../SP500_Data_sampled", "SP500_Features_sampled")
    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")