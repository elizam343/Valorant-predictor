# Valorant Esports Kill Predictor

An end-to-end machine learning pipeline that predicts how many kills a professional Valorant player will get per map, and whether they'll go over or under a given kill line.

It covers the full data lifecycle: scraping 52,000+ pro match records, storing them in SQLite, engineering 25 features, training gradient boosting models, and running a daily prediction pipeline that tracks real-world results.

**Tech stack:** Python · SQL / SQLite · scikit-learn · pandas · NumPy · BeautifulSoup · requests

---

## Results

| Model | Task | Performance (held-out test set) |
|---|---|---|
| Gradient Boosting Regressor | Predict kills per map | MAE = 3.93 kills, R² = 0.307 |
| Gradient Boosting Classifier | Predict over/under a kill line | 71.4% accuracy, AUC = 0.784 |

**Live tracking:** the model has been tested on real kill lines, and every pick is logged to `bet_results.csv`. The early live sample is small (33 picks: 15 correct, 18 incorrect) and below the offline accuracy. That gap is why the test set uses synthetic historical lines. I'm working on closing it with calibration and significance testing (`calibration_audit.py`, `significance_test.py`).

---

## How it works

```
vlr.gg ──► Scraper ──► SQLite databases ──► Feature engineering ──► Model training
                         │                                              │
                         └──────────────► Daily prediction pipeline ◄───┘
                                                  │
                                         Results tracker (CSV)
```

1. **Scraping.** `Scraper/results_scraper.py` paginates through vlr.gg match results and resumes from a checkpoint if it's interrupted.
2. **Storage.** There are two SQLite databases:
   - `valorant_matches.db` stores per-map match stats (kills, deaths, ACS, ADR, assists).
   - `vlr_players.db` stores career aggregate stats (rating, KPR, K/D).
3. **Features.** There are 25 features in five groups:
   - **Career:** rating, ACS, K/D, kills/assists/first kills/first deaths per round.
   - **Context:** team strength, opponent strength, opponent kills allowed per map.
   - **Form:** recent average kills, form trend, days since the last match.
   - **Head-to-head:** past performance against the same opponent.
   - **Map and agent:** player average kills on the map, agent role, duelist flag.
4. **Training.** `model_comparison.py` compares models and saves the best one.
5. **Daily predictions.** `bet_slate.py` pulls the day's kill lines, runs both models and applies filters. A pick is skipped when:
   - the edge is under 10%,
   - the regressor and classifier disagree, or
   - the player has fewer than 15 map appearances.
6. **Tracking.** `results_tracker.py` logs actual outcomes so accuracy can be measured over time.

---

## Data quality work

Several silent data bugs were distorting the model. I found and fixed them:

- **Corrupted K/D column.** About 50% of rows stored *kills − deaths* instead of *kills ÷ deaths*, which produced negative "ratios." Fixed by computing K/D directly from raw kills and deaths.
- **1,887 players missing from training.** Players with real match history had zeroed career stats and were being filtered out. I wrote `backfill_career_stats.py` to rebuild their stats from match data.
- **Duplicate players.** Players who changed teams showed up more than once. Fixed by deduplicating before the merge.
- **A SQL aggregate crash.** A nested `AVG(SUM(...))` query failed at runtime. Fixed by removing the bad query.

---

## Project structure

```
├── Scraper/                    # vlr.gg scraper + SQLite schema
├── kill_prediction_model/
│   ├── bet_slate.py            # Daily prediction pipeline (main entry point)
│   ├── model_comparison.py     # Training + model selection
│   ├── db_data_loader.py       # Fast training data loader (SQLite)
│   ├── backfill_career_stats.py
│   ├── name_resolver.py        # Fuzzy-matches player names across sources
│   ├── results_tracker.py      # Logs actual outcomes
│   └── models/                 # Saved models
├── docs/                       # Design notes, diagnostics, backlog
└── requirements.txt
```

---

## Getting started

```bash
pip install -r requirements.txt   # Python 3.9+, no GPU required

# 1. Scrape new matches (resumes automatically)
python Scraper/results_scraper.py

# 2. Sync career stats
python kill_prediction_model/backfill_career_stats.py

# 3. Train (~5 min)
cd kill_prediction_model
python model_comparison.py --use-db --save-best

# 4. Run today's predictions
python bet_slate.py --context context.json
```

---

## Known limitations

- Career rating for backfilled players is estimated from ACS (r = 0.516 with the real VLR rating).
- First-kill and first-death rates for backfilled players use league averages.
- Historical kill lines are synthetic. Real lines are only available live.
- Players on newly promoted teams often have no match history yet.

---

## What's next

- Calibrate the classifier's probabilities and grow the live sample to test whether the edge is real.
- Add per-map first-kill data to the scraper.
- Schedule the daily pipeline automatically instead of running it by hand.
