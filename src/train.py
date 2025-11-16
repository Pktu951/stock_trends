from fetch_data import StockDataFetcher
from features import FeatureEngineer
from models import StockModels
import pandas as pd

class Trainer:
    def __init__(self, ticker, start, end):
        self.fetcher = StockDataFetcher(ticker, start, end)
        self.df = self.fetcher.fetch()
        self.fe = FeatureEngineer(self.df)
        self.df_features = self.fe.add_features()
        self.models = StockModels()

    def train_models(self):
        X_rf = self.df_features[['MA5','MA10','Return']].values
        y_rf = self.df_features['Close'].values
        self.models.train_rf(X_rf, y_rf)

        X_lstm, y_lstm = self.models.prepare_lstm_data(self.df_features['Close'])
        self.models.train_lstm(X_lstm, y_lstm, epochs=50)
        
        return self.models