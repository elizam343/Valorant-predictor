# Project Summary — Valorant Kill Line Predictor

## Current Status

Active development. Core scraping, training, and prediction pipeline is functional. Model accuracy is honest but limited (R²≈0.21); the system is most useful for identifying high-edge spots rather than betting everything it outputs.

## What's Built

### Data Collection
- `Scraper/results_scraper.py` — parallel scraper that paginates vlr.gg/matches/results with checkpointing and resume support; 5 workers by default
- `Scraper/scraper_api.py` — single-match scraper that serves match JSON on port 5003
- `Scraper/app.py` — career stats API on port 5001, pulls from vlrggapi.vercel.app
- ~38,800 raw JSON match files in `scraped_matches/`, covering 2021–2026

### ML Pipeline
- `enhanced_data_loader.py` — loads JSON files, joins career stats from DB, engineers 15 features including agent role, map-specific history, rolling form
- `gpu_trainer.py` — trains Gradient Boosting and PyTorch NN; saves models to `models/`
- `backtester.py` — temporal train/test split (configurable cutoff), evaluates OVER/UNDER accuracy by edge band with EV calculation at −110 odds
- `kill_line_fetcher.py` — SmartSyntheticLine for backtesting (weighted blend of form/map/career stats) and PrizePicks live client for real predictions

### Prediction
- `advanced_matchup_predictor.py` — loads saved GBR + NN models, generates predictions with confidence estimates and OVER/UNDER recommendations
- Auto-fetches PrizePicks line when `kill_line=0`

### Web App
- Flask app on port 5000 with form-based prediction UI
- Session authentication with `SECRET_KEY` env var

## Performance Metrics

Retrained June 2026 on 37,137 matches, 638,527 player-map rows, 15 features.

| Model | R² | MAE (kills/map) |
|---|---|---|
| Gradient Boosting | 0.208 | 4.82 |
| Neural Network | 0.153 | 5.01 |

Previous best was R²=0.094. Improvement came from fixing a silent bug (`map_name` key was `'map'` in JSON, causing `player_map_avg_kills` to collapse across all maps) and adding 3 agent-role features.

Feature importances (Gradient Boosting):
- `player_map_avg_kills`: 29.5%
- `team_strength`: 25.4%
- `player_agent_avg_kills`: 13.9%
- `opponent_team_strength`: 7.7%
- `recent_avg_kills`: 5.1%

## 15 Features

| Feature | Source |
|---|---|
| db_rating | Career stats DB |
| db_average_combat_score | Career stats DB |
| db_kill_deaths | Career stats DB |
| db_kills_per_round | Career stats DB |
| db_assists_per_round | Career stats DB |
| db_first_kills_per_round | Career stats DB |
| db_first_deaths_per_round | Career stats DB |
| team_strength | Aggregated match win rates |
| opponent_team_strength | Aggregated match win rates |
| recent_avg_kills | Rolling 5-match kill average |
| recent_avg_rating | Rolling 5-match rating average |
| player_map_avg_kills | Player's LOO avg kills on this map |
| agent_role_ordinal | 0=Sentinel, 1=Controller, 2=Initiator, 3=Duelist |
| is_duelist | Binary flag |
| player_agent_avg_kills | Player's LOO avg kills on this agent |

## Fixed Bugs (June 2026)

1. **map_name key** — JSON uses `'map'` not `'map_name'`; `player_map_avg_kills` was always computing across all maps. Fixed in `_parse_match_data()`.
2. **Agent extraction** — VLR.gg stores agent in `<img alt="Jett">`, not text. Old scraper used `td.text.strip()` which returned `''`. Fixed in `results_scraper.py`.
3. **Backtester missing agent features** — `_load_full_dataset()` wasn't calling `add_agent_features()`, causing a KeyError. Fixed.

## Remaining Limitations

1. **Synthetic kill lines** — historical backtesting uses a computed proxy for the bookmaker line; real lines include sharp money and public betting data that we cannot replicate. Expect true live accuracy to be ~1–3pp lower than backtested results.
2. **No live inference loop** — predictions are manual (CLI or web app); there is no scheduled bot.
3. **Agent data gaps** — matches scraped with the old pipeline (IDs < ~440K) may have empty agent fields, which lowers the value of `player_agent_avg_kills` for older players.
4. **Model is weak at extremes** — R²=0.21 means there is meaningful unexplained variance. High-kill matches (star fraggers in easy opponents) and low-kill matches (support players) are harder to predict.

## File Structure

```
├── Scraper/
│   ├── results_scraper.py         # Primary scraper (parallel, checkpoint)
│   ├── scraper_api.py             # Single-match scraper / Flask API
│   ├── database_schema.py         # DB schema
│   ├── app.py                     # Career stats API
│   └── db_utils.py
├── kill_prediction_model/
│   ├── enhanced_data_loader.py
│   ├── gpu_trainer.py
│   ├── advanced_matchup_predictor.py
│   ├── kill_line_fetcher.py       # SmartSyntheticLine + PrizePicks client
│   ├── backtester.py
│   └── models/
│       ├── gradient_boosting_gpu_model.pkl
│       ├── neural_network_gpu_model.pkl
│       └── gpu_training_report.json
├── web_app/
│   ├── app.py
│   └── templates/
├── scraped_matches/               # ~38,800 JSON files
└── requirements.txt
```

## Roadmap

1. **More recent data** — scraper is still running, adding 2023–2026 matches. Re-train once stable.
2. **Live betting integration** — PrizePicks client is built; need a scheduler to run predictions before each match day.
3. **Line history** — storing historical PrizePicks lines would allow true out-of-sample evaluation without synthetic proxies.
4. **Series-level target** — current target is per-map kills; a per-series (bo3 total) target might better match PrizePicks lines.
