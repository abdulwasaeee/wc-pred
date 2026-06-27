import pandas as pd

from wcpredictor.features import build_features


def test_build_features_columns():
    """build_features should output the exact feature columns we expect."""
    df = pd.read_csv("data/results.csv", parse_dates=["date"])
    feat = build_features(df)
    for col in ["elo_diff", "form_home", "form_away", "neutral", "result"]:
        assert col in feat.columns


def test_neutral_is_binary():
    """The neutral flag must only ever be 0 or 1."""
    df = pd.read_csv("data/results.csv", parse_dates=["date"])
    feat = build_features(df)
    assert set(feat["neutral"].unique()).issubset({0, 1})
