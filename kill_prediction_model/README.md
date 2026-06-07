# Kill Prediction Model

Predicts per-map kill counts for Valorant players using Gradient Boosting and a PyTorch neural network. Used for OVER/UNDER betting analysis against kill lines.

## Files

| File | Purpose |
|---|---|
| `enhanced_data_loader.py` | Loads scraped JSON matches, joins career stats, engineers 15 features |
| `gpu_trainer.py` | Trains GBR and NN models, saves to `models/` |
| `advanced_matchup_predictor.py` | Loads saved models, makes predictions for a given matchup |
| `kill_line_fetcher.py` | SmartSyntheticLine (for backtesting) + PrizePicks live client |
| `backtester.py` | Temporal evaluation with configurable train/test cutoff |
| `models/` | Saved `.pkl` files and `gpu_training_report.json` |

## Performance

Trained on 37,137 matches (638,527 player-map rows), 15 features.

- **Gradient Boosting**: R²=0.208, MAE=4.82 kills/map
- **Neural Network**: R²=0.153, MAE=5.01 kills/map

Target: `match_kills` — kills on a single map (range 1–94, mean ~13).

## Features

15 features split into career stats (from `Scraper/vlr_players.db`), match-level context, rolling form, and agent role:

- `db_rating`, `db_average_combat_score`, `db_kill_deaths`, `db_kills_per_round`, `db_assists_per_round`, `db_first_kills_per_round`, `db_first_deaths_per_round`
- `team_strength`, `opponent_team_strength`
- `recent_avg_kills`, `recent_avg_rating`
- `player_map_avg_kills` — leave-one-out average kills on this map
- `agent_role_ordinal` — 0=Sentinel, 1=Controller, 2=Initiator, 3=Duelist
- `is_duelist` — binary
- `player_agent_avg_kills` — leave-one-out average kills on this agent

## Training

```bash
python gpu_trainer.py                         # Full training
python gpu_trainer.py --limit-matches 2000    # Quick test
```

## Prediction

```python
from advanced_matchup_predictor import AdvancedMatchupPredictor, MatchupContext

predictor = AdvancedMatchupPredictor()
result = predictor.predict_matchup(MatchupContext(
    player_name="aspas",
    player_team="MIBR",
    opponent_team="FUR",
    tournament="VCT Champions",
    series_type="bo3",
    maps=["Ascent", "Bind"],
    kill_line=16.5,
    agent="Jett"
))
# result.recommendation: OVER / UNDER / UNSURE
# result.edge_pct: model edge vs kill line
```

## Backtesting

```bash
python backtester.py --cutoff 2024-01-01
```

Trains on pre-cutoff data only, evaluates on post-cutoff matches. Reports accuracy and EV by edge band at −110 odds. Kill lines are synthetic (SmartSyntheticLine) since historical bookmaker lines are unavailable.

## Windows Setup

1. Clone repo and install: `pip install -r requirements.txt`
2. Place `valorant_matches.db` in `Scraper/`
3. Run `python kill_prediction_model/gpu_trainer.py`

## Troubleshooting

- **Missing `models/` directory**: create it with `mkdir kill_prediction_model/models`
- **NaN predictions**: player not in career stats DB — model falls back to dataset medians
- **KeyError on agent columns**: ensure `backtester.py` calls `loader.add_agent_features(df)`
