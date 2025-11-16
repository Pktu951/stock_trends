from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

class StockModels:
    def __init__(self):
        self.rf_model = RandomForestRegressor()
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_lstm_data(self, data, n_steps=5):
        scaled = self.scaler.fit_transform(data.values.reshape(-1,1))
        X, y = [], []
        for i in range(n_steps, len(scaled)):
            X.append(scaled[i-n_steps:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def build_lstm(self, n_steps=5):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        return model

    def train_rf(self, X, y):
        self.rf_model.fit(X, y)

    def predict_rf(self, X):
        return self.rf_model.predict(X)

    def train_lstm(self, X, y, epochs=50, batch_size=32):
        if self.lstm_model is None:
            self.build_lstm(X.shape[1])
        self.lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_lstm(self, X):
        if self.lstm_model is None:
            raise ValueError("LSTM model is not built yet!")
        return self.lstm_model.predict(X)
