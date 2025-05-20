import pandas as pd
import numpy as np
from datetime import datetime
import requests
from sklearn.model_selection import train_test_split

from get_CRSP_data import *
from helper_functions import *

STOCK_START_DATE = '2015-05-01'
STOCK_END_DATE = '2025-05-01'

def get_SP500_tickers_and_split():
    # scrape the sockanalysis website for all the stocks in the S&P 500 and store it into a dataframe
    website_url = "https://stockanalysis.com/list/sp-500-stocks/"
    html = requests.get(website_url).content
    sp_companies = pd.read_html(html)[0][["Symbol", "Stock Price", "Revenue"]]

    # Sample 50% of the stocks for now
    sp_companies_sampled = sp_companies.sample(frac=0.5, random_state=23)

    # Want to stratify based on the stock price (want to be fair with giving the prediction model good and not so good companies)
    sp_companies_sampled['label'] = pd.qcut(sp_companies["Stock Price"], q=8, labels=False)

    # Use a stratied split based on the stock price
    train_stocks, test_val_stocks = train_test_split(
        sp_companies_sampled,
        train_size=0.75,
        stratify=sp_companies_sampled["label"],
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

    return sp_companies_sampled

def get_stock_data(db: wrds.Connection, stock_tickers: list[str], output_path: str):

    # For each ticker we want get the adjusted price data
    for ticker in stock_tickers:
        get_CRSP_ret_data_with_dates(db=db, ticker_symbol=ticker, save_data=True, 
                                 start_date=STOCK_START_DATE, end_date=STOCK_END_DATE, output_path=output_path)


if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    # Scrape the web for the S&P 500 companies
    sp500_data = get_SP500_tickers_and_split()

    # Initialize the connection to wrds
    db = initialize_db()

    # For each ticker get the stock data
    tickers = sp500_data["Symbol"].to_list()
    output_path = "SP500_data_sampled_ret"
    initialize_directory(output_path)
    get_stock_data(db=db, stock_tickers=tickers, output_path=output_path)

    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")