import os

import pandas as pd
from fastapi import FastAPI

from wcpredictor.predict import predict_match
from wcpredictor.train import train_and_save

DATA_PATH = "data/results.csv"
MODEL_PATH = "model.pkl"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

app = FastAPI(title="World Cup Predictor")


@app.on_event("startup")
def ensure_model():
    """On a fresh server, fetch data and train the model if they're missing."""
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    if not os.path.exists(MODEL_PATH):
        train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH)


@app.get("/")
def home():
    return {"message": "World Cup Predictor API. Try /predict?home=Brazil&away=France"}


@app.get("/predict")
def predict(home: str, away: str, neutral: bool = True):
    return predict_match(home, away, neutral=neutral)
