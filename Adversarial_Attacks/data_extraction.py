import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pytorch_forecasting
from pathlib import Path
import os


def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)


SAMPLE_LENGTH = 500


def get_attack_data(
    data_path: str,
    split_path: str,
    output_path: str,
    subsample: bool = False,
    random: bool = False,
):
    print("hi")

    test_set = np.load(split_path, allow_pickle=True).item()["test"]
    print(len(test_set))

    test_data = []
    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv" and entry.name.split(".csv")[0] in test_set:
            df = pd.read_csv(entry)
            if subsample:
                if random:
                    random_start = np.random.randint(0, len(df) - SAMPLE_LENGTH)
                    df = df.loc[random_start : random_start + SAMPLE_LENGTH, :]
                else:
                    df = df.tail(SAMPLE_LENGTH)

            test_data.append(df)
            df.to_csv(f"{output_path}/{entry.name}")


OUTPUT_PATH_FULL = "SP500_AttackData_Full"

initialize_directory(OUTPUT_PATH_FULL)
get_attack_data("SP500_Data", "training_split.npy", OUTPUT_PATH_FULL)
