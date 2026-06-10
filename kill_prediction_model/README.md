# kill_prediction_model

Daily bet slate generation and model training for Valorant kill line predictions.

---

## Quick Start

```bash
# Run today's slate
python bet_slate.py --context context.json

# Retrain the model (~5 min)
python model_comparison.py --use-db --save-best

# Log a result after a game
python results_tracker.py result <player> <actual_kills>
```

---

## File Reference

### Active / Use These

| File | What it does |
|---|---|
| `bet_slate.py` | **Main daily script.** Fetches live PrizePicks lines, builds features, runs both models, applies filters, outputs BET / NO BET / Weak / Skip |
| `model_comparison.py` | **Training script.** Trains all model types, prints comparison, saves best regression + classifier to `models/` |
| `db_data_loader.py` | Fast training data loader — reads from `valorant_matches.db` directly (use with `--use-db`) |
| `enhanced_data_loader.py` | Slow training data loader — reads all 52k JSON files from `scraped_matches/` |
| `backfill_career_stats.py` | Syncs `vlr_players.db` career stats from match data for players the VLR scraper missed. Run after any bulk scraping session. Also auto-runs at `model_comparison.py` startup. |
| `results_tracker.py` | Logs actual outcomes to `bet_results.csv` |
| `kill_line_fetcher.py` | PrizePicks live API client + SmartSyntheticLine for training |
| `name_resolver.py` | Fuzzy-matches PrizePicks player names to VLR DB names, writes `name_aliases.json` |
| `context.json` | **Update this daily.** Maps each active player to their team and opponent for today's matches |
| `name_aliases.json` | PP → VLR name corrections. Auto-populated by `name_resolver.py`, hand-edit to fix wrong suggestions |
| `bet_results.csv` | Running log of all picks, predictions, and actual outcomes |

### Legacy / Do Not Use

| File | Status |
|---|---|
| `gpu_trainer.py` | Old training script. Superseded by `model_comparison.py` |
| `advanced_matchup_predictor.py` | Old inference script. Superseded by `bet_slate.py` |

---

## Model Details

Two GBR models work in tandem:

**Regressor** — predicts raw kill count
- R² = 0.307, MAE = 3.93 kills/map
- Saved as `models/best_regression_model.pkl`

**Classifier** — predicts P(OVER) directly
- 71.4% accuracy, AUC = 0.784
- Saved as `models/best_classifier_model.pkl`
- Gets 2 extra features: `synthetic_line` and `player_hit_rate_at_line`

### 23 Base Features

**Career (from `vlr_players.db`):**
- `db_rating`, `db_average_combat_score`, `db_kill_deaths`
- `db_kills_per_round`, `db_assists_per_round`
- `db_first_kills_per_round`, `db_first_deaths_per_round`

**Match context:**
- `team_strength`, `opponent_team_strength`, `opponent_kills_allowed_per_map`
- `avg_rounds_vs_opponent` — estimated rounds from (total_map_kills / 6.0) history by team pairing

**Rolling form:**
- `recent_avg_kills` — 10-map rolling avg (shift 1)
- `recent_avg_kills_3` — 3-map rolling avg (shift 1, faster recency signal)
- `recent_avg_rating` — 10-map rolling avg of match rating
- `form_slope` — linear slope of last 5 maps
- `days_since_last_match` — rest days, clipped 0–30

**H2H:**
- `h2h_avg_kills` — leave-one-out avg kills vs this specific opponent
- `h2h_data_exists` — 1 if real H2H data exists, 0 if first-time matchup

**Map / Agent:**
- `player_map_avg_kills` — leave-one-out avg kills on this map
- `kill_std` — player's historical kill standard deviation
- `agent_role_ordinal` — 0=Sentinel, 1=Controller, 2=Initiator, 3=Duelist
- `is_duelist` — binary
- `player_agent_avg_kills` — leave-one-out avg kills on this agent

### Bet Filters

| Filter | Trigger | Action |
|---|---|---|
| Base-rate | OVER + DB hit_rate < 45% | NO BET, edge = 0 |
| Conflict | Reg says UNDER, clf says OVER | NO BET, edge × 0.5 |
| Weak signal | Edge < 10% | Listed separately, don't bet |
| Skip | < 15 map appearances | Not shown |

Break-even at −110 odds: **52.4%**

---

## context.json Format

Update this every day before running the slate. Maps each PrizePicks player name to their team and opponent:

```json
{
  "_note": "Today's matches: TeamA vs TeamB | TeamC vs TeamD",
  "playerone": {"team": "Team A", "opponent": "Team B"},
  "playertwo": {"team": "Team B", "opponent": "Team A"}
}
```

Player names must match PrizePicks exactly (lowercase). If a player shows as "no data" in the slate, run `name_resolver.py` to check for a name alias needed.

---

## Training

```bash
# Recommended — fast, uses DB
python model_comparison.py --use-db --save-best

# Full JSON dataset — slow (~2h), slightly more data
python model_comparison.py --save-best

# Quick dev run
python model_comparison.py --use-db --limit-matches 5000
```

The `--use-db` path uses `db_data_loader.py` which reads `Scraper/valorant_matches.db` via a single SQL query. Much faster than reading 52k JSON files. The kdr column in that DB is corrupt (stores kd_diff for ~50% of rows) — `db_data_loader.py` works around this by computing `kills/deaths` directly.

---

## Logging Results

```bash
python results_tracker.py result kingg 24        # kingg got 24 kills
python results_tracker.py result blowz 31        # blowz got 31 kills
python results_tracker.py summary                # win/loss breakdown
```

Results are appended to `bet_results.csv`.

---

## After a Scraping Session

When new matches are scraped into `valorant_matches.db`, new players won't have career stats in `vlr_players.db`. Run the backfill first:

```bash
python backfill_career_stats.py
python model_comparison.py --use-db --save-best
```

The backfill computes career averages (ACS, KD, KPR, APR) from match data and inserts them for any player not already in `vlr_players.db`. The VLR rating is approximated as `0.002382 × ACS + 0.4872` (r=0.516 vs real). First kills/deaths per round default to league means (0.092 / 0.107).

---

## Where We Left Off (2026-06-07)

- **Retrain killed mid-run** — needs to be rerun: `python model_comparison.py --use-db --save-best`
- **Today's results not yet logged** — wait for games to finish, then log via `results_tracker.py`
- **Today's picks**: keiko UNDER 34.5 (19.1% edge), brawk UNDER 31.5 (16.3%), kingg OVER 27.5 (15.3%), sociablee UNDER 27.5 (15.1%)
- **Keiko note**: avg kills 15.68/map, only 1.9% career hit rate above 34.5. Line is more than 2× her career average. Strong UNDER.
