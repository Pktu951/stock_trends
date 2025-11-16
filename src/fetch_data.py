import yfinance as yf
import pandas as pd

class StockDataFetcher:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end

    def fetch(self):
        df = yf.download(self.ticker, start=self.start, end=self.end)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        return df