# World Cup Predictor — Product Design

## Background
- Users: football fans & bracket-pickers wanting a data-driven view; us as learners.
- Goals: a credible, numbers-backed forecast of the tournament.
- Pains: existing predictions are gut-feel punditry or paywalled, unexplained odds.

## Value Proposition
- Product: predicts match outcomes, simulates the tournament for each team's title %.
- Alleviates: replaces guessing with transparent probabilities.
- Advantages: free, explainable, updates with real results.

## Objectives
- Predict match outcome probabilities. [FOCUS]
- Simulate the 48-team bracket to get title probabilities.
- Serve results via an API.

## Solution
- Core features: match win/draw/loss probabilities [FOCUS]; tournament simulation; API.
- Integrations: input = two teams; later ingest live results.
- Alternatives: plain Elo (no ML); bookmaker odds — rejected, defeats the learning goal.
- Constraints: probabilities sum to 1; low latency; calibrated outputs.
- Out-of-scope: player/injury modeling, live in-game prediction, betting.

## Feasibility
- Data: historical international results + FIFA rankings (free, Kaggle).
- Open question: enough signal (form, ranking, home advantage) to predict well?


# World Cup Predictor — Systems Design

## Data
- Training: historical international results + FIFA rankings.
- Production: 2026 fixtures + live results.
- Features: recent form, Elo rating, ranking gap, home advantage.
- Label: match outcome (home win / draw / away win).
- Leakage guard: split by DATE, never shuffle. Train on past, test on newer.

## Metrics
- Primary: log loss + Brier score (reward calibrated probabilities).
- Sanity: accuracy vs. a baseline.

## Evaluation
- Offline: holdout set of past matches the model never saw.
- Online: compare predictions vs. real 2026 results as they happen.

## Modeling (manual before ML)
1. Rule baseline: higher-ranked team wins.
2. Simple statistical model.
3. XGBoost — each step benchmarked against the previous.

## Inference
- Batch: pre-compute the 104 fixtures + tournament simulation.
- Online: API serves any matchup on demand.

## Feedback
- Real results become new training data; watch for drift; retrain.