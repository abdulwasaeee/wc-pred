from wcpredictor.predict import predict_match


def test_probabilities_sum_to_one():
    """The three outcome probabilities must add up to 1 (100%)."""
    result = predict_match("Brazil", "France")
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01, f"Probabilities summed to {total}, not 1"


def test_stronger_team_favoured():
    """A much stronger team should get a higher win probability."""
    result = predict_match("Brazil", "Scotland")  # Brazil far stronger
    assert result["home_win"] > result["away_win"], "Brazil should be favoured over Scotland"


def test_unknown_team_raises():
    """Asking about a team not in the data should raise a clear error."""
    import pytest

    with pytest.raises(ValueError):
        predict_match("Brazil", "Wakanda")  # not a real team
