import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random

def initialize_directory(path: str) -> None:
    """Create the output folder at path if it does not exist, or empty it if it exists."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.mkdir(path)




def filter_dataset(data_path: str, output_path: str):

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            df = pd.read_csv(entry)
            # If the stock has less than 600 days of data then ignore it (should be moved to collect_data or when we make the split)
            if len(df) < 2434:
                continue
            else:
                df.to_csv(f"{output_path}/{entry.name.split(".csv")[0]}.csv", index=False)


def create_log_return_plots(data_path: str, output_path: str, subsample = False):

    for entry in Path(data_path).iterdir():
        if entry.suffix == ".csv":
            # If the stock has less than 600 days of data then ignore it (should be moved to collect_data or when we make the split)
            adjprc = pd.read_csv(entry)["adjprc"].to_numpy()

            # do log returns, this way we can easily inverse and get adjprc when we have our second discriminator (need initial price though)
            log_returns = np.log(adjprc[1:] / adjprc[:-1])

            if subsample:
                start = random.randint(0, len(log_returns) - 500)
                log_returns = log_returns[start : start+500]

            plt.plot(log_returns)
            plt.xlabel('Time (Days)')
            plt.ylabel('Log Returns')
            plt.title(f'Log Returns for {entry.name.split(".csv")[0]}')
            plt.savefig(f"{output_path}/{entry.name.split(".csv")[0]}.png")
            plt.close()
            print(f"Finished {entry}")

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # output_path = 'SP500_Filtered'
    # initialize_directory(output_path)
    # filter_dataset(data_path="SP500_Data", output_path=output_path)

    output_path = 'LogReturnPlotsSubsample'
    initialize_directory(output_path)
    create_log_return_plots(data_path='SP500_Filtered', output_path=output_path, subsample=True)

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")