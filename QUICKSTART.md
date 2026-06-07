# Quick Start

## Setup

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
# Full training (37K+ matches, ~2h on CPU)
python kill_prediction_model/gpu_trainer.py

# Quick test with a subset
python kill_prediction_model/gpu_trainer.py --limit-matches 2000
```

Saves models to `kill_prediction_model/models/`.

## Make a Prediction

```python
import sys
sys.path.insert(0, 'kill_prediction_model')
from advanced_matchup_predictor import AdvancedMatchupPredictor, MatchupContext

predictor = AdvancedMatchupPredictor()

matchup = MatchupContext(
    player_name="aspas",
    player_team="MIBR",
    opponent_team="FUR",
    tournament="VCT Champions",
    series_type="bo3",
    maps=["Ascent", "Bind"],
    kill_line=16.5,   # kills per map (typical range: 10–22)
    agent="Jett"      # optional but improves accuracy
)

result = predictor.predict_matchup(matchup)
print(f"Predicted: {result.predicted_kills:.1f} kills/map")
print(f"Kill line: {result.kill_line:.1f}")
print(f"Recommendation: {result.recommendation}")  # OVER / UNDER / UNSURE
print(f"Edge: {result.edge_pct:.1f}%")
```

**Important**: `kill_line` is kills per map, not kills per round. A typical player averages ~13 kills/map. PrizePicks lines are usually set between 13.5 and 20.5.

Set `kill_line=0` to auto-fetch the current PrizePicks line for the player (requires an active projection).

## Run from Command Line

```bash
cd kill_prediction_model
python advanced_matchup_predictor.py
```

Edit the `main()` block in `advanced_matchup_predictor.py` to change the matchup context.

## Scrape New Matches

```bash
# Scrape vlr.gg results pages (auto-resumes from checkpoint)
python Scraper/results_scraper.py

# With options
python Scraper/results_scraper.py --workers 5 --max-pages 200 --start-page 1
```

Match JSON files are saved to `scraped_matches/`. The scraper is polite (1.5–2.5s delay between pages, 0.5–1.0s between match pages).

## Backtest the Model

```bash
# Train on pre-2024 data, evaluate on 2024+ matches
python kill_prediction_model/backtester.py --cutoff 2024-01-01
```

Outputs OVER/UNDER accuracy by edge band and expected value at −110 odds. The edge is measured against a synthetic kill line (50% recent form, 30% map avg, 20% career KPR + opponent adjustment).

## Web App

```bash
cd web_app
python app.py
# Open http://localhost:5000
```

Set `SECRET_KEY` env var for non-dev deployments:
```bash
SECRET_KEY=your-secret-here python app.py
```

## Troubleshooting

**Player not found / low accuracy**: The player may not be in the career stats database. The model falls back to dataset averages for missing features, which reduces accuracy.

**KeyError on agent features**: Make sure you're using the latest `backtester.py` which calls `loader.add_agent_features(df)`.

**PrizePicks returns nothing**: The player has no active projection right now. Provide a `kill_line` manually.

**Scraper blocked**: VLR.gg rate-limits aggressive scrapers. The default delay settings are polite; if you hit 429s, increase `PAGE_DELAY` in `results_scraper.py`.
