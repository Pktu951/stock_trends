import numpy as np

class Predictor:
    def __init__(self, models, df_features):
        self.models = models
        self.df_features = df_features

    def predict_rf(self):
        X_rf = self.df_features[['MA5','MA10','Return']].values
        return self.models.predict_rf(X_rf)

    def predict_lstm(self):
        X_lstm, _ = self.models.prepare_lstm_data(self.df_features['Close'])
        return self.models.predict_lstm(X_lstm)
