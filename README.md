# Valorant Kill Line Predictor

A machine learning system that predicts per-map kill counts for Valorant esports players and recommends OVER/UNDER bets against PrizePicks kill lines.

---

## Where We Left Off (2026-06-07)

**Active pipeline is `bet_slate.py`** — not the old `gpu_trainer.py` / `advanced_matchup_predictor.py`. Those are legacy.

**Last session work:**
- Fixed 3 data quality leaks (see Data Quality section below)
- Backfilled career stats for 1,887 previously-missing players
- Model: GBR Classifier 71.4% acc, AUC=0.784 / Regressor R²=0.307, MAE=3.93

**Pending — retrain was killed mid-run:**
```bash
cd kill_prediction_model
python model_comparison.py --use-db --save-best
```
This will take ~5-10 minutes using the DB loader. Run this before the next slate.

**Today's slate (2026-06-07) matches:**
- Vitality vs FUT Esports
- NRG vs Leviatán
- Dragon Ranger vs Xi Lai Gaming
- Global Esports vs FULL SENSE

Results not yet logged. After games finish:
```bash
python kill_prediction_model/results_tracker.py result <player> <actual_kills>
```

---

## Daily Workflow

```bash
# 1. Update context.json with today's matchups (team/opponent per player)
#    File: kill_prediction_model/context.json

# 2. Run the slate
cd kill_prediction_model
python bet_slate.py --context context.json

# 3. Log results after games
python results_tracker.py result kingg 24
python results_tracker.py result blowz 18
```

Results are saved to `kill_prediction_model/bet_results.csv`.

---

## Architecture

```
valorant-kill-line-predictor/
├── Scraper/
│   ├── valorant_matches.db        # Match-by-match stats (kills, deaths, ACS per map)
│   ├── vlr_players.db             # Career aggregate stats (rating, KPR, KD, etc.)
│   ├── results_scraper.py         # Scraper: paginates vlr.gg/matches/results
│   ├── database_schema.py         # Inserts scraped matches into valorant_matches.db
│   └── db_utils.py                # Connection helper for vlr_players.db
├── kill_prediction_model/
│   ├── bet_slate.py               # MAIN DAILY SCRIPT — fetches lines, runs model, outputs picks
│   ├── model_comparison.py        # Training script — run to retrain
│   ├── db_data_loader.py          # Fast training data loader from valorant_matches.db
│   ├── enhanced_data_loader.py    # Slow training data loader from 52k JSON files
│   ├── backfill_career_stats.py   # Syncs vlr_players.db from match data (run after scraping)
│   ├── results_tracker.py         # Logs actual outcomes to bet_results.csv
│   ├── kill_line_fetcher.py       # PrizePicks live API client
│   ├── name_resolver.py           # Fuzzy-matches PP player names to VLR DB names
│   ├── context.json               # TODAY'S matchups — update this daily
│   ├── name_aliases.json          # PP name → VLR canonical name mappings
│   ├── bet_results.csv            # Running log of all picks + outcomes
│   └── models/                    # Saved model .pkl files
├── scraped_matches/               # 52k+ raw JSON match files
└── README.md
```

---

## Two Databases — Critical Distinction

| Database | What it stores | How it's populated |
|---|---|---|
| `Scraper/valorant_matches.db` | Per-map match stats (kills, deaths, ACS, ADR, assists) | `database_schema.py` / `results_scraper.py` |
| `Scraper/vlr_players.db` | Career aggregate stats (rating, KPR, KD, etc.) | VLR player-page scraper (separate process) |

**These are independent** — vlr_players.db is NOT auto-populated when new matches come in. After any bulk scraping session, run:

```bash
cd kill_prediction_model
python backfill_career_stats.py
```

This computes career averages from match data and inserts them for any player not already in vlr_players.db. It also runs automatically at the start of `model_comparison.py`.

---

## Model

Two models are used together at inference time:

| Model | Task | Performance |
|---|---|---|
| GBR Regressor | Predicts kill count | R²=0.307, MAE=3.93 kills/map |
| GBR Classifier | Predicts OVER/UNDER probability | 71.4% accuracy, AUC=0.784 |

**23 features** (regression) / **25 features** (classification adds `synthetic_line` + `player_hit_rate_at_line`):
- Career: `db_rating`, `db_average_combat_score`, `db_kill_deaths`, `db_kills_per_round`, `db_assists_per_round`, `db_first_kills_per_round`, `db_first_deaths_per_round`
- Context: `team_strength`, `opponent_team_strength`, `opponent_kills_allowed_per_map`, `avg_rounds_vs_opponent`
- Form: `recent_avg_kills`, `recent_avg_kills_3`, `recent_avg_rating`, `form_slope`, `days_since_last_match`
- H2H: `h2h_avg_kills`, `h2h_data_exists`
- Map/Agent: `player_map_avg_kills`, `kill_std`, `agent_role_ordinal`, `is_duelist`, `player_agent_avg_kills`

**Bet filters applied at inference:**
- `player_hit_rate_at_line < 45%` and direction is OVER → **NO BET** (base-rate filter)
- Regressor says UNDER but classifier says OVER → **NO BET** (conflict filter)
- Edge < 10% → **Weak signal** (don't bet)
- < 15 map appearances → **Skipped**

Break-even at −110 odds: 52.4%

---

## Training

```bash
cd kill_prediction_model

# Fast (recommended) — reads from valorant_matches.db, ~5 min
python model_comparison.py --use-db --save-best

# Thorough (slow) — reads all 52k JSON files, ~2 hours
python model_comparison.py --save-best

# Quick dev check
python model_comparison.py --use-db --limit-matches 5000
```

---

## Scraping New Matches

```bash
# Scrape from vlr.gg/matches/results — auto-resumes from checkpoint
python Scraper/results_scraper.py

# After scraping, sync career stats DB
python kill_prediction_model/backfill_career_stats.py

# Then retrain
python kill_prediction_model/model_comparison.py --use-db --save-best
```

---

## Data Quality Issues (Fixed 2026-06-07)

1. **`kdr` column corrupt in `player_match_stats`** — `database_schema.py` stored `kd_diff` (kills−deaths) as primary value with kd_ratio as fallback, so ~50% of rows had negative "KDR". Fixed in `db_data_loader.py` to compute kills/deaths directly.

2. **Career DB coverage gap** — 1,887 players with real match history had 0.0 for all career features and were being silently excluded from training (enhanced_data_loader filters `db_rating > 0`). Fixed by `backfill_career_stats.py`.

3. **Duplicate player names in `vlr_players.db`** — same player listed under multiple teams. Fixed in `_load_career_stats()` via deduplication before merge.

4. **SQL crash in rounds query** — nested `AVG(SUM(...))` in `bet_slate.py` PlayerCache threw `misuse of aggregate function`. Fixed by removing the bad query.

---

## Known Limitations

- `db_rating` for backfilled players is estimated from ACS (r=0.516 vs real VLR composite rating) — not exact but much better than 0.0
- `first_kills_per_round` / `first_deaths_per_round` for backfilled players use league mean (0.092 / 0.107) — per-map FK data not stored in the DB
- Kill lines are synthetic for historical data — real PrizePicks lines are fetched live at inference
- Dragon Ranger / Xi Lai Gaming players routinely have 0 DB appearances (debut skips)
- `p.rating` in JSON match files is the per-match VLR rating; `db_rating` in vlr_players.db is a career aggregate — different scales, both used in different parts of the pipeline

---

## Installation

```bash
pip install -r requirements.txt
# Python 3.9+, no GPU required
```

Place both SQLite databases in `Scraper/` before running.
