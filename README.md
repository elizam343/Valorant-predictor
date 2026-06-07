# Valorant Kill Line Predictor

A machine learning system that predicts per-map kill counts for Valorant esports players, built to support OVER/UNDER betting analysis against bookmaker kill lines.

## Overview

Data is scraped from VLR.gg, stored as JSON match files, and used to train regression models that predict how many kills a player will get on a given map. The system compares predictions against kill lines to generate OVER/UNDER recommendations with an edge estimate.

## Architecture

```
valorant-kill-line-predictor/
├── Scraper/
│   ├── results_scraper.py         # Parallel scraper (vlr.gg/matches/results)
│   ├── scraper_api.py             # Single-match scraper / Flask API (port 5003)
│   ├── database_schema.py         # SQLite schema and migration helpers
│   ├── app.py                     # Career stats API (port 5001, vlrggapi data)
│   └── db_utils.py                # Database utilities
├── kill_prediction_model/
│   ├── enhanced_data_loader.py    # Loads JSON files, engineers features
│   ├── gpu_trainer.py             # Trains GBR + PyTorch NN models
│   ├── advanced_matchup_predictor.py  # Loads saved models, makes predictions
│   ├── kill_line_fetcher.py       # SmartSyntheticLine + PrizePicks live client
│   ├── backtester.py              # Temporal train/test split evaluation
│   └── models/                    # Saved .pkl model files + training report
├── web_app/
│   ├── app.py                     # Flask web interface (port 5000)
│   └── templates/
├── scraped_matches/               # Raw JSON match files (~38,800 files)
├── requirements.txt
└── README.md
```

## Model Performance

Retrained June 2026 on 37,137 matches (638,527 player-map rows), 15 features.

| Model | R² | MAE |
|---|---|---|
| Gradient Boosting (primary) | 0.208 | 4.82 kills/map |
| PyTorch Neural Network | 0.153 | 5.01 kills/map |

Target variable: kills per map (range 1–94, mean ~13). Kill lines are typically 13–20 kills/map.

Top features by importance (Gradient Boosting):
1. `player_map_avg_kills` — 29.5%
2. `team_strength` — 25.4%
3. `player_agent_avg_kills` — 13.9%
4. `opponent_team_strength` — 7.7%
5. `recent_avg_kills` — 5.1%

Note: R² of 0.2 reflects genuine difficulty — kills per map are noisy (round count varies, side distributions vary, opponent tier varies). The model is useful when it generates a >10% edge against the kill line.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. PyTorch CPU is fine for inference; GPU speeds up training only.

## Training

```bash
# Full training (~2h on CPU, uses all JSON files)
python kill_prediction_model/gpu_trainer.py

# Quick test run
python kill_prediction_model/gpu_trainer.py --limit-matches 2000
```

Models are saved to `kill_prediction_model/models/`.

## Prediction

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
    kill_line=16.5,   # kills per map (NOT kills per round)
    agent="Jett"      # optional; improves prediction
)

result = predictor.predict_matchup(matchup)
print(f"Predicted kills/map: {result.predicted_kills:.1f}")
print(f"Kill line: {result.kill_line:.1f}")
print(f"Recommendation: {result.recommendation}")  # OVER / UNDER / UNSURE
print(f"Edge: {result.edge_pct:.1f}%")
```

Set `kill_line=0` to auto-fetch a live line from PrizePicks (when the player has an active projection).

## Scraping New Matches

```bash
# Scrape from vlr.gg/matches/results — auto-resumes from checkpoint
python Scraper/results_scraper.py

# Options
python Scraper/results_scraper.py --workers 5 --max-pages 100
```

Checkpoint is saved to `Scraper/.results_scraper_checkpoint.json`. The scraper writes JSON files to `scraped_matches/` and also inserts into `Scraper/valorant_matches.db`.

## Backtesting

```bash
# Evaluate OVER/UNDER accuracy with a temporal train/test split
python kill_prediction_model/backtester.py --cutoff 2024-01-01
```

The backtester trains only on pre-cutoff data, then measures accuracy on post-cutoff matches against synthetic kill lines (50% recent form, 30% map avg, 20% career KPR + opponent adjustment). Live bookmaker lines are not available historically, so the synthetic line is an approximation.

## Web App

```bash
cd web_app
python app.py
# Open http://localhost:5000
```

The web app provides a form-based interface for single-player predictions. Authentication is session-based with a secret key set via the `SECRET_KEY` environment variable.

## Data Sources

- **Match data**: VLR.gg match result pages scraped via BeautifulSoup
- **Career stats**: vlrggapi.vercel.app (used for `db_rating`, `db_kills_per_round`, etc.)
- **Live kill lines**: PrizePicks public projections API (Valorant league ID 36)

## Known Limitations

- Kill lines are synthetic for historical backtesting — real bookmaker lines include public betting patterns that the model cannot see.
- Agent data is missing for older matches scraped before agent extraction was fixed.
- `player_map_avg_kills` had a silent bug (used 'map_name' key instead of 'map') for the first ~34K matches; those have been re-patched.
- No live inference pipeline — predictions are generated manually or via the web app.

## License

Educational and research use only. Respect VLR.gg's terms of service when scraping.
