from train import Trainer
from predict import Predictor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from db import init_db, get_connection

def predict_future_rf(models, df_features, n_days=30):
    """Predykcje przyszłych cen RandomForest"""
    close_series = df_features['Close'].values.copy()
    future_preds = []

    for _ in range(n_days):
        last_ma5 = float(close_series[-5:].mean())
        last_ma10 = float(close_series[-10:].mean())
        last_return = float((close_series[-1] - close_series[-2]) / close_series[-2])

        last_X = np.array([[last_ma5, last_ma10, last_return]], dtype=float)
        pred = models.predict_rf(last_X)[0]
        future_preds.append(pred)
        close_series = np.append(close_series, pred)

    return np.array(future_preds)


def predict_future_lstm(models, df_features, n_days=30):
    """Predykcje przyszłych cen LSTM"""
    close_series = df_features['Close'].values.copy()
    future_preds = []

    scaler = models.scaler
    scaled_close = scaler.fit_transform(close_series.reshape(-1,1)).flatten()

    n_steps = 5
    for _ in range(n_days):
        last_sequence = scaled_close[-n_steps:]
        X = last_sequence.reshape(1, n_steps, 1)
        pred_scaled = models.predict_lstm(X)[0,0]
        future_preds.append(pred_scaled)
        scaled_close = np.append(scaled_close, pred_scaled)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
    return future_preds

def main():
    ticker = "AAPL"
    start = "2025-07-01"
    end = "2025-09-01"

    trainer = Trainer(ticker, start, end)
    models = trainer.train_models()
    print("Modele wytrenowane!")

    predictor = Predictor(models, trainer.df_features)
    rf_preds = predictor.predict_rf()
    
    X_lstm, _ = models.prepare_lstm_data(trainer.df_features['Close'])
    lstm_preds_scaled = models.predict_lstm(X_lstm)
    lstm_preds = models.scaler.inverse_transform(lstm_preds_scaled).flatten()

    print("Predykcje RandomForest:", rf_preds[-5:])
    print("Predykcje LSTM:", lstm_preds[-5:])

    future_days = 30
    rf_future = predict_future_rf(models, trainer.df_features.copy(), n_days=future_days)
    lstm_future = predict_future_lstm(models, trainer.df_features.copy(), n_days=future_days)

    future_index = pd.date_range(trainer.df_features.index[-1] + pd.Timedelta(days=1),
                                 periods=future_days)

    plt.figure(figsize=(12,6))
    plt.plot(trainer.df_features.index, trainer.df_features['Close'], label='Rzeczywiste', color='black')
    plt.plot(trainer.df_features.index[-len(rf_preds):], rf_preds, label='RandomForest', color='blue')
    plt.plot(trainer.df_features.index[-len(lstm_preds):], lstm_preds, label='LSTM', color='red')
    plt.plot(future_index, rf_future, '--', label='RF przyszłe', color='blue')
    plt.plot(future_index, lstm_future, '--', label='LSTM przyszłe', color='red')

    plt.legend()
    plt.title(f"Przewidywanie cen akcji {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.ylim(150, 300)
    plt.grid(True)
    plt.show()
    results_df = pd.DataFrame({
        "date": future_index,
        "rf_future": rf_future,
        "lstm_future": lstm_future
    })

    results_df.to_csv("predictions.csv", index=False)
    print("Zapisano predykcje do predictions.csv")


if __name__ == "__main__":
    main()
