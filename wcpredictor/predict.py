import joblib
import pandas as pd

from wcpredictor.features import latest_ratings


def predict_match(
    home: str,
    away: str,
    neutral: bool = True,
    data_path: str = "data/results.csv",
    model_path: str = "model.pkl",
) -> dict:
    """Predict a match outcome from two team names."""
    raw = pd.read_csv(data_path, parse_dates=["date"])
    elo, form = latest_ratings(raw)

    for team in (home, away):
        if team not in elo:
            raise ValueError(f"Unknown team: {team!r}")

    X = pd.DataFrame(
        [
            {
                "elo_diff": elo[home] - elo[away],
                "form_home": form[home],
                "form_away": form[away],
                "neutral": int(neutral),
            }
        ]
    )
    model = joblib.load(model_path)
    proba = model.predict_proba(X)[0]
    return {cls: round(float(p), 3) for cls, p in zip(model.classes_, proba)}
