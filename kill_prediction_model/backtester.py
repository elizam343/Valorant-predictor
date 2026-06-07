#!/usr/bin/env python3
"""
Temporal backtester for the kill line predictor.

Workflow
--------
1. Load all match data with dates preserved.
2. Split by cutoff date (train = before, test = after).
3. Fit a fresh gradient boosting model on training data only.
4. For each test row generate a synthetic kill line = player's pre-match
   rolling average (the best proxy for what a bookmaker would set).
5. Predict kills for every test row.
6. Measure OVER/UNDER accuracy bucketed by percentage edge.
7. Compute calibration: at each confidence band, are we right that % of the time?
8. Compute expected value at −110 odds across thresholds.
9. Print a betting report.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from enhanced_data_loader import EnhancedDataLoader
from kill_line_fetcher import SmartSyntheticLine

CUTOFF      = pd.Timestamp('2022-07-01')
AVG_ROUNDS  = 18.5          # Typical Valorant map round count (used to convert kpr → kills)
BREAKEVEN   = 0.5238        # −110 odds break-even rate
ODDS_WIN    = 100           # Profit per $110 bet at −110
ODDS_LOSS   = 110           # Loss per bet at −110

_smart_line = SmartSyntheticLine()

FEATURE_COLUMNS = [
    'db_rating', 'db_average_combat_score', 'db_kill_deaths',
    'db_kills_per_round', 'db_assists_per_round',
    'db_first_kills_per_round', 'db_first_deaths_per_round',
    'team_strength', 'opponent_team_strength',
    'recent_avg_kills', 'recent_avg_rating',
    'player_map_avg_kills',
    # Agent role features
    'agent_role_ordinal',
    'is_duelist',
    'player_agent_avg_kills',
]
TARGET = 'match_kills'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_kill_line(row: pd.Series) -> float:
    """Smart weighted kill line: 50% recent form + 30% map avg + 20% career KPR + opponent adj."""
    return _smart_line.compute(row)


def pct_edge(predicted: float, kill_line: float) -> float:
    """Absolute percentage difference between prediction and kill line."""
    if kill_line <= 0:
        return 0.0
    return abs(predicted - kill_line) / kill_line * 100


def ev_per_bet(accuracy: float) -> float:
    """Expected value of a $110 flat bet at −110 odds."""
    return accuracy * ODDS_WIN - (1 - accuracy) * ODDS_LOSS


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(self, cutoff: pd.Timestamp = CUTOFF, limit_matches: int = None):
        self.cutoff = cutoff
        self.limit  = limit_matches

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _load_full_dataset(self) -> pd.DataFrame:
        """Return one row per player-map with features + date + actual kills."""
        loader = EnhancedDataLoader()
        matches = loader.load_scraped_matches(limit=self.limit)
        df = loader.create_training_dataset(matches)
        df = loader.create_kill_prediction_features(df)
        df = loader.add_rolling_features(df)
        df = loader.add_agent_features(df)
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        return df

    def _split(self, df: pd.DataFrame):
        mask  = df['match_date'] < self.cutoff
        train = df[mask].copy()
        test  = df[~mask & df['match_date'].notna()].copy()

        # Fix temporal leakage in player_map_avg_kills:
        # The add_rolling_features() method uses leave-one-out across ALL data,
        # meaning test rows include "future" map appearances.
        # Replace with training-data-only map averages.
        train_map_avg = (
            train.groupby(['player_name', 'map_name'])['match_kills']
            .mean()
            .reset_index()
            .rename(columns={'match_kills': 'player_map_avg_kills_fixed'})
        )
        test = test.merge(train_map_avg, on=['player_name', 'map_name'], how='left')
        # Fallback: player's overall train-set average, then dataset mean
        player_overall = (
            train.groupby('player_name')['match_kills']
            .mean()
            .reset_index()
            .rename(columns={'match_kills': 'player_overall_avg'})
        )
        test = test.merge(player_overall, on='player_name', how='left')
        test['player_map_avg_kills'] = (
            test['player_map_avg_kills_fixed']
            .fillna(test['player_overall_avg'])
            .fillna(train['match_kills'].mean())
        )
        test = test.drop(columns=['player_map_avg_kills_fixed', 'player_overall_avg'],
                         errors='ignore')

        # Same fix for training set (its map avg already excludes current row via
        # leave-one-out, but some rows share future-match counts across the boundary;
        # recompute from pre-cutoff data only for consistency).
        train = train.drop(columns=['player_map_avg_kills'], errors='ignore')
        grp_map   = train.groupby(['player_name', 'map_name'])['match_kills']
        map_sum   = grp_map.transform('sum')
        map_count = grp_map.transform('count')
        train['player_map_avg_kills'] = (map_sum - train['match_kills']) / (map_count - 1)
        p_avg = train.groupby('player_name')['match_kills'].transform('mean')
        train['player_map_avg_kills'] = train['player_map_avg_kills'].fillna(p_avg)

        return train, test

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows with career stats and real kills; fill sparse features."""
        sub = df[FEATURE_COLUMNS + [TARGET, 'match_date']].copy()
        sub = sub[(sub['db_rating'] > 0) & (sub[TARGET] > 0)]
        sub = sub.fillna(sub.median(numeric_only=True))
        return sub

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _fit(self, train: pd.DataFrame):
        scaler = StandardScaler()
        X = scaler.fit_transform(train[FEATURE_COLUMNS])
        y = train[TARGET].values
        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )
        model.fit(X, y)
        r2_train = model.score(X, y)
        print(f'  Train R² (in-sample):  {r2_train:.4f}')
        return model, scaler

    def _predict(self, model, scaler, test: pd.DataFrame) -> np.ndarray:
        X = scaler.transform(test[FEATURE_COLUMNS])
        return model.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, test: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        df = test.copy()
        df['predicted_kills'] = preds
        df['kill_line']       = df.apply(synthetic_kill_line, axis=1)
        df['pct_edge']        = df.apply(
            lambda r: pct_edge(r['predicted_kills'], r['kill_line']), axis=1
        )
        df['predicted_over']  = df['predicted_kills'] > df['kill_line']
        df['actual_over']     = df[TARGET] > df['kill_line']
        df['correct']         = df['predicted_over'] == df['actual_over']
        return df

    def _calibration_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accuracy + EV by percentage-edge bucket."""
        bins   = [0, 5, 10, 15, 20, 30, 100]
        labels = ['0–5%', '5–10%', '10–15%', '15–20%', '20–30%', '>30%']
        df['edge_band'] = pd.cut(df['pct_edge'], bins=bins, labels=labels, right=False)

        rows = []
        for band in labels:
            sub = df[df['edge_band'] == band]
            if len(sub) == 0:
                continue
            acc = sub['correct'].mean()
            rows.append({
                'edge_band':  band,
                'n_bets':     len(sub),
                'accuracy':   round(acc, 4),
                'ev_per_bet': round(ev_per_bet(acc), 2),
                'profitable': acc >= BREAKEVEN,
            })
        return pd.DataFrame(rows)

    def _threshold_sweep(self, df: pd.DataFrame) -> pd.DataFrame:
        """At each minimum edge threshold, what is accuracy + EV?"""
        rows = []
        for threshold in range(0, 35, 5):
            sub = df[df['pct_edge'] >= threshold]
            if len(sub) < 30:
                break
            acc = sub['correct'].mean()
            rows.append({
                'min_edge_%':    threshold,
                'n_bets':        len(sub),
                'pct_of_total':  round(len(sub) / len(df) * 100, 1),
                'accuracy':      round(acc, 4),
                'ev_per_100':    round(ev_per_bet(acc), 2),
                'profitable':    acc >= BREAKEVEN,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        print('=' * 60)
        print(' VALORANT KILL LINE PREDICTOR — BACKTEST')
        print(f' Train: before {self.cutoff.date()}  |  Test: on/after')
        print('=' * 60)

        print('\n[1/4] Loading match data...')
        full_df = self._load_full_dataset()
        train_raw, test_raw = self._split(full_df)
        print(f'  Raw train rows: {len(train_raw):,}')
        print(f'  Raw test rows:  {len(test_raw):,}')

        print('\n[2/4] Cleaning and splitting...')
        train = self._clean(train_raw)
        test  = self._clean(test_raw)
        print(f'  Clean train rows: {len(train):,}')
        print(f'  Clean test rows:  {len(test):,}')

        if len(test) < 50:
            print('ERROR: too few test rows — widen the date range or use more matches.')
            return

        print('\n[3/4] Training on pre-cutoff data only...')
        model, scaler = self._fit(train)

        preds = self._predict(model, scaler, test)
        oos_r2  = r2_score(test[TARGET].values, preds)
        oos_mse = mean_squared_error(test[TARGET].values, preds)
        print(f'  Out-of-sample R²:  {oos_r2:.4f}')
        print(f'  Out-of-sample MAE: {np.mean(np.abs(preds - test[TARGET].values)):.3f} kills/map')

        print('\n[4/4] Measuring betting accuracy...')
        results = self._evaluate(test, preds)

        # Naive baselines for comparison
        dataset_mean = train[TARGET].mean()
        results['naive_pred']   = dataset_mean
        results['naive_correct'] = (
            (results['naive_pred'] > results['kill_line']) == results['actual_over']
        )
        naive_acc        = results['naive_correct'].mean()
        always_over_acc  = results['actual_over'].mean()

        # Overall stats
        overall_acc = results['correct'].mean()
        print(f'\n  Kill line: smart blend (50% recent form, 30% map avg, 20% career KPR + opp. adj.)')
        print(f'    Range: {results["kill_line"].min():.1f} – {results["kill_line"].max():.1f} kills/map')
        print(f'    Mean:  {results["kill_line"].mean():.1f}  Median: {results["kill_line"].median():.1f}')

        print(f'\n  ── Accuracy comparison ──────────────────────────────')
        print(f'  Always-predict-OVER:         {always_over_acc:.1%}')
        print(f'  Naive (predict dataset mean): {naive_acc:.1%}')
        print(f'  Our model:                   {overall_acc:.1%}  (+{overall_acc-naive_acc:.1%} vs naive)')
        print(f'  Bookmaker break-even:        {BREAKEVEN:.1%}')
        beats = '✓ BEATS' if overall_acc >= BREAKEVEN else '✗ below'
        print(f'  Status: {beats} break-even overall')
        print(f'\n  ⚠ CAVEAT: kill lines are synthetic (weighted blend of player signals).')
        print(f'  Real bookmakers also incorporate public betting patterns and sharp money.')
        print(f'  Expect true edge to be ~1–3pp lower against live PrizePicks/Underdog lines.')

        # Calibration table
        cal = self._calibration_report(results)
        print('\n── Accuracy by edge band ──────────────────────────────────')
        print(f"  {'Edge':>8}  {'N bets':>7}  {'Accuracy':>9}  {'EV / $110 bet':>14}  {'Profitable?':>11}")
        print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*14}  {'-'*11}")
        for _, row in cal.iterrows():
            prof = '  YES' if row['profitable'] else '   no'
            print(f"  {row['edge_band']:>8}  {row['n_bets']:>7,}  {row['accuracy']:>8.1%}  "
                  f"${row['ev_per_bet']:>+12.2f}  {prof:>11}")

        # Threshold sweep
        sweep = self._threshold_sweep(results)
        print('\n── Threshold sweep: bet only when model edge ≥ X% ─────────')
        print(f"  {'Min edge':>9}  {'N bets':>7}  {'% of pool':>9}  "
              f"{'Accuracy':>9}  {'EV / $110':>10}  {'Profitable?':>11}")
        print(f"  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*11}")
        for _, row in sweep.iterrows():
            prof = '  YES' if row['profitable'] else '   no'
            print(f"  {row['min_edge_%']:>8}%  {row['n_bets']:>7,}  "
                  f"{row['pct_of_total']:>8.1f}%  {row['accuracy']:>8.1%}  "
                  f"${row['ev_per_100']:>+8.2f}  {prof:>11}")

        # Recommendation
        profitable_sweeps = sweep[sweep['profitable']]
        print('\n── Recommendation ─────────────────────────────────────────')
        if profitable_sweeps.empty:
            print('  Model does not beat break-even at any tested threshold.')
            print('  Do not use for betting without fresher data (2023-2026).')
        else:
            best = profitable_sweeps.iloc[0]
            print(f"  Best threshold: edge ≥ {best['min_edge_%']:.0f}%")
            print(f"    Accuracy:  {best['accuracy']:.1%}  (need ≥ {BREAKEVEN:.1%})")
            print(f"    EV:        ${best['ev_per_100']:+.2f} per $110 bet")
            print(f"    Pool size: {best['n_bets']:,} bets ({best['pct_of_total']:.1f}% of matchups)")
            print()
            print('  NOTE: results are based on 2021-2023 data which is now stale.')
            print('  Expect accuracy to improve after scraping 2023-2026 matches.')
        print('=' * 60)

        return results, cal, sweep


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff',  default='2022-07-01', help='Train/test split date (YYYY-MM-DD)')
    parser.add_argument('--limit',   type=int, default=None, help='Limit matches loaded (for testing)')
    args = parser.parse_args()

    bt = Backtester(cutoff=pd.Timestamp(args.cutoff), limit_matches=args.limit)
    bt.run()
