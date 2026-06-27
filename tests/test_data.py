import pandas as pd


def test_no_negative_scores():
    """Football scores can never be negative."""
    df = pd.read_csv("data/results.csv")
    played = df.dropna(subset=["home_score", "away_score"])
    assert (played["home_score"] >= 0).all()
    assert (played["away_score"] >= 0).all()


def test_expected_columns_exist():
    """The dataset must have the columns our pipeline relies on."""
    df = pd.read_csv("data/results.csv")
    needed = {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
    assert needed.issubset(df.columns), f"Missing columns: {needed - set(df.columns)}"
