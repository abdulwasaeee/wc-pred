import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="World Cup Predictor")
model = joblib.load("model.pkl")


@app.get("/predict")
def predict(elo_diff: float, form_home: float = 1.5, form_away: float = 1.5, neutral: int = 1):
    X = pd.DataFrame(
        [
            {
                "elo_diff": elo_diff,
                "form_home": form_home,
                "form_away": form_away,
                "neutral": neutral,
            }
        ]
    )
    proba = model.predict_proba(X)[0]
    return {cls: round(float(p), 3) for cls, p in zip(model.classes_, proba)}
