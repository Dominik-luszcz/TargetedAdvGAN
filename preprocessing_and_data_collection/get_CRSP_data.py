from datetime import datetime
import wrds
import sys

'''
Iniitialize the WRDS database
'''

def initialize_db():
    db = wrds.Connection()
    return db


'''
Given a ticker symbol, get the daily stock price from CRSP (WRDS)

I only have access to the crsp db and not the crspa that is discussed on the website
'''
def get_CRSP_data(db: wrds.Connection, ticker_symbol: str, save_data: bool = False, output_path: str = None):

    # If multiple companies with same permno we take the newest company
    try:
        desired_permno = db.raw_sql(f"""
                                SELECT permno, namedt
                                FROM crsp.msenames
                                WHERE ticker = '{ticker_symbol}'
                                ORDER BY namedt DESC
                                LIMIT 1
                                """)["permno"].item()
    except:
        print(f"\n Could not find ticker symbol {ticker_symbol}, please try again. \n")
        return
    
    # Given the ticker get the raw price (prc) and the adjusted price based on the formula on the WRDS website
    df = db.raw_sql(f"""
                    SELECT permno, '{ticker_symbol}' AS ticker, prc AS rawprc, ABS(prc) / cfacpr AS adjprc, date AS date
                    FROM crsp.dsf
                    WHERE permno = '{desired_permno}'""")
    
    if save_data:
        if output_path == None:
            df.to_csv(f"{ticker_symbol}_price_data.csv", index=False)
        else:
            df.to_csv(f"{output_path}/{ticker_symbol}.csv", index=False)

    return df

def get_CRSP_data_with_dates(db: wrds.Connection, ticker_symbol: str, 
                             start_date: str, end_date: str, save_data: bool = False, output_path: str = None):

    # If multiple companies with same permno we take the newest company
    try:
        desired_permno = db.raw_sql(f"""
                                SELECT permno, namedt
                                FROM crsp.msenames
                                WHERE ticker = '{ticker_symbol}'
                                ORDER BY namedt DESC
                                LIMIT 1
                                """)["permno"].item()
    except:
        print(f"\n Could not find ticker symbol {ticker_symbol}, please try again. \n")
        return
    
    # Given the ticker get adjusted price based on the formula on the WRDS website between certain dates
    df = db.raw_sql(f"""
                    SELECT permno, '{ticker_symbol}' AS ticker, ABS(prc) / cfacpr AS adjprc, date AS date
                    FROM crsp.dsf
                    WHERE permno = '{desired_permno}'
                        AND date BETWEEN '{start_date}' AND '{end_date}'""")
    
    if save_data:
        if output_path == None:
            df.to_csv(f"{ticker_symbol}_price_data.csv", index=False)
        else:
            df.to_csv(f"{output_path}/{ticker_symbol}.csv", index=False)

    return df

if __name__ == '__main__':
    t1 = datetime.now()
    print(f"Started job at {t1}")

    try:
        ticker = sys.argv[1]
    except:
        ticker = 'AAPL'

    db = initialize_db()
    
    data = get_CRSP_data(db, ticker, save_data=True)
    
    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")