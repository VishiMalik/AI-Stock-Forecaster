import yfinance as yf
import pandas as pd

def fetch_history(ticker: str, period = "3y", interval = "1d"):
  t = yf.Ticker(ticker)
  df = t.history(period = period, interval = interval)
  df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
  df.dropna(inplace = True)
  return df

if __name = "__main__":
   df = fetch_history("AAPL", period = "5y")
   print(df.tail())

