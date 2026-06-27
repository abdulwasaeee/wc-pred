from functools import lru_cache
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from wcpredictor.features import latest_ratings

_DATA = "data/results.csv"
_MODEL = "model.pkl"

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Sweden", "Tunisia", "Japan"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def simulate_group(teams, rng):
    """Play a group round-robin; return teams ranked best-to-worst, plus points."""
    pts = {t: 0 for t in teams}
    for home, away in combinations(teams, 2):  # every pair plays once
        outcome = simulate_match(home, away, rng)
        if outcome == "home_win":
            pts[home] += 3
        elif outcome == "away_win":
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], ELO.get(t, 1500)), reverse=True)
    return ranked, pts


def simulate_group_stage(rng):
    """Simulate all 12 groups; return the 32 teams that advance."""
    winners, runners_up, thirds = [], [], []
    for teams in GROUPS.values():
        ranked, pts = simulate_group(teams, rng)
        winners.append(ranked[0])
        runners_up.append(ranked[1])
        thirds.append((ranked[2], pts[ranked[2]]))
    best_thirds = [
        t for t, _ in sorted(thirds, key=lambda x: (x[1], ELO.get(x[0], 1500)), reverse=True)[:8]
    ]
    return winners + runners_up + best_thirds


# load once at import — recomputing per match would be far too slow
_raw = pd.read_csv(_DATA, parse_dates=["date"])
ELO, FORM = latest_ratings(_raw)
MODEL = joblib.load(_MODEL)


@lru_cache(maxsize=None)
def match_probs(home: str, away: str, neutral: bool = True) -> dict:
    """Model's win/draw/loss probabilities for one matchup."""
    X = pd.DataFrame(
        [
            {
                "elo_diff": ELO.get(home, 1500) - ELO.get(away, 1500),
                "form_home": FORM.get(home, 1.5),
                "form_away": FORM.get(away, 1.5),
                "neutral": int(neutral),
            }
        ]
    )
    return dict(zip(MODEL.classes_, MODEL.predict_proba(X)[0]))


def simulate_match(home: str, away: str, rng, knockout: bool = False) -> str:
    """Play one match by sampling from the model's probabilities."""
    p = match_probs(home, away)
    outcome = rng.choice(
        ["home_win", "draw", "away_win"],
        p=[p["home_win"], p["draw"], p["away_win"]],
    )
    if knockout and outcome == "draw":
        # a knockout can't end level — resolve by strength (extra time + pens)
        ph, pa = ELO.get(home, 1500), ELO.get(away, 1500)
        outcome = "home_win" if rng.random() < 1 / (1 + 10 ** ((pa - ph) / 400)) else "away_win"
    return outcome


def simulate_knockout(teams, rng):
    """Single-elimination from a list of teams down to one champion."""
    bracket = list(teams)
    rng.shuffle(bracket)
    while len(bracket) > 1:
        nxt = []
        for i in range(0, len(bracket), 2):
            outcome = simulate_match(bracket[i], bracket[i + 1], rng, knockout=True)
            nxt.append(bracket[i] if outcome == "home_win" else bracket[i + 1])
        bracket = nxt
    return bracket[0]


def simulate_tournament(rng):
    """One full tournament: group stage feeding the knockout."""
    return simulate_knockout(simulate_group_stage(rng), rng)


def title_odds(n=10000, seed=0):
    """Run n tournaments; return a Counter of each team's title count."""
    from collections import Counter

    rng = np.random.default_rng(seed)
    wins = Counter()
    for _ in range(n):
        wins[simulate_tournament(rng)] += 1
    return wins
