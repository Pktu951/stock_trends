import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def add_features(self):
        df = self.df.copy()
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df.dropna(inplace=True)
        self.df = df
        return df