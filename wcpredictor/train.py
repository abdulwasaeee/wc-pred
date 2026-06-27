import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wcpredictor.features import build_features

FEATURES = ["elo_diff", "form_home", "form_away", "neutral"]


def train_model(raw: pd.DataFrame) -> LogisticRegression:
    """Build features and fit the champion model on all played matches."""
    feat = build_features(raw)
    model = LogisticRegression(max_iter=1000).fit(feat[FEATURES], feat["result"])
    return model


def train_and_save(data_path: str = "data/results.csv", model_path: str = "model.pkl") -> None:
    """End-to-end: read raw data, train, and write the model to disk."""
    raw = pd.read_csv(data_path, parse_dates=["date"])
    model = train_model(raw)
    joblib.dump(model, model_path)
    print(f"Trained on {len(raw)} rows -> saved {model_path}")
