from fastapi import FastAPI
from train import Trainer
from predict import Predictor
import uvicorn

app = FastAPI()

ticker = "AAPL"
start = "2023-01-01"
end = "2025-01-01"

trainer = Trainer(ticker, start, end)
models = trainer.train_models()

predictor = Predictor(models, trainer.df_features)

@app.get("/predict/rf")
def predict_rf():
    preds = predictor.predict_rf()
    return {"predictions": preds.tolist()}

@app.get("/predict/lstm")
def predict_lstm():
    preds = predictor.predict_lstm()
    return {"predictions": preds.flatten().tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
