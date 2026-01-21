import yfinance as yf
import pandas as pd

# Define tickers
tickers = {
    'Silver': 'SI=F',
    'Gold': 'GC=F',
    'Dollar Index': 'DX-Y.NYB',
    '10Y Treasury': '^TNX'
}

print("Attempting to fetch data from Yahoo Finance...")

for name, ticker in tickers.items():
    print(f"\nFetching {name} ({ticker})...")
    try:
        data = yf.Ticker(ticker)
        # Fetch last 5 days
        hist = data.history(period="5d")
        if not hist.empty:
            print(f"Success! Last price: {hist['Close'].iloc[-1]:.2f}")
            print(hist[['Close']].tail())
        else:
            print(f"Failed: No data found for {ticker}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

print("\nCheck complete.")
