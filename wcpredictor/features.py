from collections import defaultdict, deque

import pandas as pd


def build_features(raw: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    """Turn raw match rows into model features (Elo, form, neutral).

    Computes each team's rating and recent form going INTO each match,
    using only prior results — so it's leakage-safe by construction.
    """
    played = (
        raw.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    )

    def result(h, a):
        return "home_win" if h > a else "away_win" if h < a else "draw"

    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=5))
    rows = []

    for r in played.itertuples():
        h, a = r.home_team, r.away_team
        eh, ea = elo[h], elo[a]
        fh = sum(form[h]) / len(form[h]) if form[h] else 1.5
        fa = sum(form[a]) / len(form[a]) if form[a] else 1.5
        res = result(r.home_score, r.away_score)

        rows.append(
            {
                "date": r.date,
                "home_team": h,
                "away_team": a,
                "elo_diff": eh - ea,
                "form_home": fh,
                "form_away": fa,
                "neutral": int(str(r.neutral).upper() == "TRUE"),
                "result": res,
            }
        )

        s_home = 1.0 if res == "home_win" else 0.5 if res == "draw" else 0.0
        exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
        elo[h] = eh + k * (s_home - exp_home)
        elo[a] = ea + k * ((1 - s_home) - (1 - exp_home))
        form[h].append(3 if res == "home_win" else 1 if res == "draw" else 0)
        form[a].append(3 if res == "away_win" else 1 if res == "draw" else 0)

    return pd.DataFrame(rows)


def latest_ratings(raw: pd.DataFrame, k: int = 30):
    """Return each team's CURRENT Elo and recent-form after all played matches."""
    played = (
        raw.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    )
    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=5))

    for r in played.itertuples():
        h, a = r.home_team, r.away_team
        res = (
            "home_win"
            if r.home_score > r.away_score
            else "away_win"
            if r.home_score < r.away_score
            else "draw"
        )
        s_home = 1.0 if res == "home_win" else 0.5 if res == "draw" else 0.0
        exp_home = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        elo[h] += k * (s_home - exp_home)
        elo[a] += k * ((1 - s_home) - (1 - exp_home))
        form[h].append(3 if res == "home_win" else 1 if res == "draw" else 0)
        form[a].append(3 if res == "away_win" else 1 if res == "draw" else 0)

    avg_form = {t: (sum(v) / len(v) if v else 1.5) for t, v in form.items()}
    return dict(elo), avg_form
