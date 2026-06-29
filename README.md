# World Cup 2026 Predictor

A machine learning system that predicts international football match outcomes and simulates the 2026 FIFA World Cup to estimate each team's chance of winning the title.

**Live API:** https://wc-pred.onrender.com
**Try a prediction:** https://wc-pred.onrender.com/predict?home=Brazil&away=France
**Interactive docs:** https://wc-pred.onrender.com/docs

## What it does

- Rates every national team using an Elo system built from ~49,000 historical matches (1872–present)
- Predicts win / draw / loss probabilities for any matchup with a calibrated logistic regression model
- Simulates the full 48-team tournament 10,000 times (Monte Carlo) to produce title odds for every team

## How it works

Raw match results feed an Elo rating per team. Those ratings, plus recent form and home/neutral venue, become features for a logistic regression model that outputs calibrated match probabilities. A Monte Carlo simulation replays the tournament thousands of times to turn single-match odds into each team's probability of lifting the trophy.

## Tech

Python, pandas, scikit-learn, FastAPI, pytest, Ruff, GitHub Actions (CI), deployed on Render.