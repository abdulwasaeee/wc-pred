import typer

from wcpredictor.predict import predict_match
from wcpredictor.simulate import title_odds
from wcpredictor.train import train_and_save

app = typer.Typer(help="World Cup match-outcome predictor.", no_args_is_help=True)


@app.command()
def train(data_path: str = "data/results.csv", model_path: str = "model.pkl"):
    """Train the model on historical matches and save it to disk."""
    train_and_save(data_path=data_path, model_path=model_path)


@app.command()
def predict(home: str, away: str, neutral: bool = True):
    """Predict a match outcome between two teams."""
    result = predict_match(home, away, neutral=neutral)
    for outcome, prob in result.items():
        print(f"  {outcome:9s} {prob * 100:.1f}%")


@app.command()
def odds(n: int = 10000):
    """Estimate each team's chance of winning the 2026 World Cup."""
    wins = title_odds(n=n)
    for team, count in wins.most_common(20):
        print(f"  {team:16s} {count / n * 100:5.1f}%")


if __name__ == "__main__":
    app()
