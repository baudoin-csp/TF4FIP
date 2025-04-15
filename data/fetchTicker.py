import argparse

import pandas as pd
import yfinance as yf

# python fetchTicker.py --session_start pd.Timestamp("2023-04-07 00:00:00", tz="America/Chicago") --session_end pd.Timestamp("2025-04-07 00:00:00", tz="America/Chicago") --output_path data_2.csv --ticker "ES=F"


def fetch(ticker="ES=F", frequency="1h", session_start=None, session_end=None):

    # Fetch historical data for the specific trading session at 1-hour intervals
    data = yf.download(
        tickers=ticker,
        interval=frequency,
        start=session_start,
        end=session_end,
        prepost=True,
    )
    if frequency == "1h":
        data.index = data.index.tz_convert("America/Chicago")
        
    data.columns = data.columns.droplevel("Ticker")
    # Reset index to obtain index values instead of only dates
    data.reset_index(inplace=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Name of the ticker")
    parser.add_argument(
        "--frequency",
        type=str,
        default="1h",
        choices=["1h", "1d"],
        help="Frequency of the fetched data",
    )
    parser.add_argument(
        "--session_start", type=str, required=True, help="Start of the trading session."
    )
    parser.add_argument(
        "--session_end", type=str, required=True, help="End of the trading session"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data_es_future.csv",
        help="The name of the destination file.",
    )
    args = parser.parse_args()

    session_start = pd.Timestamp(args.session_start, tz="America/Chicago")
    session_end = pd.Timestamp(args.session_end, tz="America/Chicago")

    df = fetch(
        ticker=args.ticker,
        frequency=args.frequency,
        session_start=session_start,
        session_end=session_end,
    )
    df.to_csv(args.output_path)
