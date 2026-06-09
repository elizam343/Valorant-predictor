# Kill-Line Predictor — Improvement Backlog

> **Canonical tracker is now GitHub Issues** — epic [#8](https://github.com/elizam343/Valorant-predictor/issues/8).
> Actionable items have their own issues (#2–#7). This file is a readable snapshot; update the issues, not this.

Status: ✅ done · 🔜 next · ⬜ todo · ⛔ blocked.

**The one constraint that frames everything:** current best model is MAE **4.63 kills/map**
(LightGBM, R²=0.389). Over a 2-map total that's ~6.5-kill error vs. the ~4-kill edges we bet —
**our error still exceeds our edge.** Every item below is ultimately about closing that gap
*or* proving (via the betting corpus) whether we have an edge at all.

Standing record: **0 wins / 3 losses settled** (killua, kingg, sociablee — last two need
actuals logged). Slates skew heavily OVER → systematic bias, not edge.

---

## ✅ Done (this session + recent)
- ✅ **Train/serve leak fix** — chronological split + causal hit-rate (old R²/AUC were inflated)
- ✅ **Distributional model + ¼-Kelly** (`kill_distribution.py`) — models the 2-map SUM, real P(over)
- ✅ **line_history corpus + backtester** — infra to validate (data-starved until results accrue)
- ✅ **Market shrinkage λ=0.5** — pulls hot μ̂ toward the line, kills all-OVER slates (`--market-weight`)
- ✅ **Role-conditioned variance v1** — bootstrap from on-agent history for flex players
      (`select_kill_hist`); would have AVOIDED the kingg loss
- ✅ **rating_form_slope** — efficiency-trajectory feature (rating trend, not kills). Retrained:
      MAE 4.66→4.63, AUC .760→.764. Real (importance #11/24) but marginal.

## 🔜 NOW — free / no new data (do as a batch)
- 🔜 **L1 training objective** — XGB `reg:absoluteerror`, LGBM `regression_l1`. Models currently
      default to L2 (optimizes the MEAN); MAE is minimized by the MEDIAN. Directly targets our
      metric. Likely > the 0.03 from rating_form_slope. **One line each.**
- 🔜 **Residual analysis script** — bucket holdout MAE by player / favored-vs-dog / blowout-vs-grind
      / role / map / recent-form. Finds *measured* weaknesses instead of guessing what to build next.
      Also check: is mean(pred) > mean(actual) on holdout? If so, recentering = free MAE.
- 🔜 **Ensemble blend** — XGB and LGBM are tied; averaging two near-tied models usually shaves MAE.

## ⬜ Tier 2 — cheap features from data we already have
- ⬜ **Recency / time-decay weighting** — old maps currently weighted equally with recent ones, in
      career means AND the bootstrap. Exponentially weight recent. Captures form + meta shifts.
- ⬜ **Strength-of-schedule adjustment** — normalize a player's kill history by opponent quality
      ("kills above expectation"), not raw average inflated by games vs. weak teams.
- ⬜ **Mismatch signal** — feed `team_strength − opponent_team_strength` (and/or ratio) explicitly;
      the gap predicts blowouts, and blowouts crater kills. (Currently only fed separately.)
- ⬜ **Team kill-environment / pace** — does this team play fast high-total-kill maps or slow ones?
- ⬜ **Interaction features** — e.g. `is_duelist × expected_rounds` (duelists scale with rounds more).

## ⬜ Tier 3 — structural / bigger lift
- ⬜ **Player random-effect / hierarchical model** — player identity is a huge driver GBR relearns
      every time. Explicit per-player intercept (mixed-effects, or OUT-OF-FOLD target encoding of
      player_id) could be one of the biggest single MAE cuts. Must be leak-safe.
- ⬜ **Count-aware loss** — kills are count data; try Poisson/Tweedie objective.
- ⬜ **Overtime / forfeit handling** — OT maps inflate kills, forfeits deflate; fat tails hurt MAE.
- ⬜ **Quantile regression** — for distribution/betting, not just the mean.

## ⬜ Round-count track (the recurring #2-feature theme) — GATED ON DATA
We have NO real round data: DB has no score cols; JSON `total_score`/`round_results` are null in
100% of sampled matches; can't be derived algebraically (deaths≈kills; ACS/ADR/KAST are per-round →
circular). Current `avg_rounds_vs_opponent` = `total_kills ÷ 6` proxy. `scraper_api.py:130-150`
ALREADY parses VLR's round timeline — the bulk scraper just didn't use it.
- ⬜ **Stage 0 — smoke test** (~30 min): does `.scoreboard-rounds` still match live VLR HTML?
- ⛔ **Stage 1 — re-scrape** `total_score` for recent / active-player matches (hours; rate-limited)
- ⛔ **Stage 2a — feature swap**: replace `÷6` proxy with TRUE rounds, retrain, measure ΔMAE
- ⛔ **Stage 2b — KPR reformulation** (the real prize): predict KPR × expected_rounds instead of
      kills directly. KPR is role/round-robust → lower relative error; blowouts mechanically lower
      predicted kills, attacking the over-bias at the source instead of patching with λ.

## ⬜ Validation / process — the ACTUAL ceiling on everything
- ⬜ **Fill the betting corpus** — log every line, settle every result daily. #1 priority overall:
      without it, no heuristic (λ, shift_cap, Kelly, MIN_AGENT_*) can be tuned and no edge proven.
- ⬜ **Settle kingg + sociablee** — actuals still not entered (0/3 record incomplete).
- ⬜ **A/B hist_source calibration** (agent vs career) once corpus has settled results.
- ⬜ **Calibration check** — does P(over)=63% actually hit ~63%?
- ⬜ **CLV** — re-log lines near match start to measure closing-line value (best skill indicator).
- ⬜ **Tune heuristics from data** — λ (try ~0.3), shift_cap, Kelly fraction, MIN_AGENT_* once corpus fills.

## ⬜ Hygiene / tech debt
- ⬜ **Consolidate FEATURE_COLS** — currently 5 copies (bet_slate, model_comparison, enhanced_data_loader
      local list, db_data_loader, backtester). Train/serve skew landmine; bit us adding rating_form_slope.
- ⬜ **Drop/fix `is_duelist`** — 0.0 importance (dead; role captured via agent_role_ordinal).
- ⬜ **`--use-db` now incompatible** with rating_form_slope (DB lacks rating) — fix or remove the path.
- ⬜ **Tests** — only the `kill_distribution` self-test exists. Add coverage for cache/features/logging.
- ⬜ **Automate results logging** — manual entry is why the corpus is empty after multiple slates.
