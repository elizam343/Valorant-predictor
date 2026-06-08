#!/usr/bin/env python3
"""
Model comparison — trains every regression AND classification model on the
same kill data and prints ranked comparison tables.

Regression   → predicts continuous kill count; metric: R², MAE
Classification → predicts OVER/UNDER a synthetic kill line; metric: accuracy, ROC-AUC, F1

Usage:
  python model_comparison.py                        # full dataset
  python model_comparison.py --limit-matches 3000   # quick dev run
  python model_comparison.py --save-best            # also saves best model to models/
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                               GradientBoostingRegressor,
                               RandomForestClassifier,
                               RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression,
                                   LogisticRegression, Ridge)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from enhanced_data_loader import EnhancedDataLoader
from db_data_loader import DBDataLoader

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('  XGBoost not installed — skipping (pip install xgboost)')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print('  LightGBM not installed — skipping (pip install lightgbm)')

MODELS_DIR = Path(__file__).parent / 'models'

# 23 features (20 previous + 3 new)
FEATURE_COLS = [
    'db_rating', 'db_average_combat_score', 'db_kill_deaths',
    'db_kills_per_round', 'db_assists_per_round',
    'db_first_kills_per_round', 'db_first_deaths_per_round',
    'team_strength',
    'opponent_team_strength',
    'opponent_kills_allowed_per_map',
    'recent_avg_kills', 'recent_avg_rating',
    'recent_avg_kills_3',        # last 3 maps (faster recency)
    'form_slope',
    'days_since_last_match',     # rest days
    'h2h_avg_kills',
    'h2h_data_exists',
    'player_map_avg_kills',
    'avg_rounds_vs_opponent',    # expected rounds from team-matchup history
    'kill_std',
    'agent_role_ordinal', 'is_duelist', 'player_agent_avg_kills',
]

# Classification gets two extra features: the kill line + player historical hit rate
CLF_EXTRA = ['synthetic_line', 'player_hit_rate_at_line']


# ---------------------------------------------------------------------------
# Synthetic line (mirrors SmartSyntheticLine from kill_line_fetcher.py)
# ---------------------------------------------------------------------------

def _make_synthetic_line(row: pd.Series) -> float:
    signals, weights = [], []
    recent = float(row.get('recent_avg_kills') or 0)
    if recent > 0:
        signals.append(recent); weights.append(0.5)
    map_avg = float(row.get('player_map_avg_kills') or 0)
    if map_avg > 0:
        signals.append(map_avg); weights.append(0.3)
    kpr = float(row.get('db_kills_per_round') or 0)
    if kpr > 0:
        signals.append(kpr * 18.5); weights.append(0.2)
    if not signals:
        return 13.0
    total = sum(weights)
    line = sum(s * w / total for s, w in zip(signals, weights))
    opp = float(row.get('opponent_team_strength') or 1.0)
    line += float(np.clip((opp - 1.0) * -0.5, -1.5, 1.5))
    return float(np.clip(line, 5.0, 35.0))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(limit_matches: int = None, use_db: bool = False) -> pd.DataFrame:
    if use_db:
        loader = DBDataLoader()
        X, y = loader.prepare_training_data()
    else:
        loader = EnhancedDataLoader()
        X, y = loader.prepare_training_data(limit_matches=limit_matches)
    if X.empty:
        raise RuntimeError('No training data loaded.')
    df = X.copy()
    df['match_kills'] = y.values
    return df


def _compute_player_hit_rates(df: pd.DataFrame) -> np.ndarray:
    """
    CAUSAL hit rate: for each row, the fraction of the player's PRIOR maps
    (strictly earlier match_date) whose kills exceeded this row's synthetic_line.

    Uses only past data, matching production — at inference time
    player_hit_rate_at_line is computed from career history available *before*
    the match. The old leave-one-out version looked at the player's future maps
    too, which leaks the target into the feature. Earliest map(s) per player have
    no prior history and default to 0.5.
    """
    kills   = df['match_kills'].values
    lines   = df['synthetic_line'].values
    players = df['player_name'].values if 'player_name' in df.columns else None

    if players is None:
        return np.full(len(df), 0.5)

    # Iterate rows in chronological order so each player's index list is
    # date-sorted; without dates we fall back to existing row order.
    if 'match_date' in df.columns:
        order = np.argsort(df['match_date'].values, kind='mergesort')  # stable
    else:
        order = np.arange(len(df))

    from collections import defaultdict
    player_indices: dict = defaultdict(list)
    for i in order:
        player_indices[players[i]].append(i)

    hit_rates = np.full(len(df), 0.5)
    for p, indices in player_indices.items():
        idx = np.array(indices)          # already sorted oldest → newest
        n   = len(idx)
        if n <= 1:
            continue
        pk = kills[idx]
        pl = lines[idx]
        # above[j, i] = pk[j] > pl[i]; keep only prior maps j < i (strict upper
        # triangle), so each row sees only kills that happened before it.
        above  = pk[:, np.newaxis] > pl[np.newaxis, :]   # (n, n)
        prior  = np.triu(above, k=1).sum(axis=0)          # hits among j < i
        counts = np.arange(n)                             # number of prior maps
        with np.errstate(invalid='ignore', divide='ignore'):
            hr = np.where(counts > 0, prior / np.maximum(counts, 1), 0.5)
        hit_rates[idx] = hr

    return hit_rates


def build_datasets(df: pd.DataFrame):
    """
    Returns:
      X_reg, y_reg   — regression (predict kill count)
      X_clf, y_clf   — classification (predict OVER/UNDER synthetic line)
      scaler_reg     — fitted StandardScaler for regression
      scaler_clf     — fitted StandardScaler for classification
    """
    # Synthetic line for each row
    df['synthetic_line'] = df.apply(_make_synthetic_line, axis=1)
    # Binary target: 1=OVER, 0=UNDER
    df['over_under'] = (df['match_kills'] > df['synthetic_line']).astype(int)

    # player_hit_rate_at_line — leave-one-out fraction of maps exceeding synthetic_line
    print('  Computing player hit rates at line (vectorised)...')
    df['player_hit_rate_at_line'] = _compute_player_hit_rates(df)

    # Class balance info
    ov = df['over_under'].mean()
    print(f'  Classification target: {ov:.1%} OVER  {1-ov:.1%} UNDER')

    # ── Regression ──────────────────────────────────────────────────────────
    X_reg = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y_reg = df['match_kills']

    # ── Classification ──────────────────────────────────────────────────────
    clf_cols = FEATURE_COLS + CLF_EXTRA
    X_clf = df[clf_cols].fillna(df[clf_cols].median())
    y_clf = df['over_under']

    # Scale (linear models need it; tree models ignore it — harmless for them)
    scaler_reg = StandardScaler().fit(X_reg)
    scaler_clf = StandardScaler().fit(X_clf)

    return X_reg, y_reg, X_clf, y_clf, scaler_reg, scaler_clf


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def regression_models(include_slow: bool = True) -> dict:
    models = {
        'Linear Regression':      LinearRegression(),
        'Ridge':                  Ridge(alpha=1.0),
        'Lasso':                  Lasso(alpha=0.1, max_iter=2000),
        'Random Forest':          RandomForestRegressor(n_estimators=200, max_depth=8,
                                                        min_samples_leaf=20, n_jobs=-1,
                                                        random_state=42),
        'Gradient Boosting':      GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                             max_depth=5, subsample=0.8,
                                                             min_samples_leaf=20, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                               max_depth=5, subsample=0.8,
                                               colsample_bytree=0.8, n_jobs=-1,
                                               random_state=42, verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                                 max_depth=5, num_leaves=31, subsample=0.8,
                                                 n_jobs=-1, random_state=42, verbose=-1)
    # sklearn GBR is ~150x slower than XGBoost for ~identical metrics — skip in fast mode.
    if not include_slow:
        models.pop('Gradient Boosting', None)
    return models


def classification_models(include_slow: bool = True) -> dict:
    models = {
        'Logistic Regression':    LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Naive Bayes':            GaussianNB(),
        'Random Forest':          RandomForestClassifier(n_estimators=200, max_depth=8,
                                                          min_samples_leaf=20, n_jobs=-1,
                                                          random_state=42),
        'Gradient Boosting':      GradientBoostingClassifier(n_estimators=400, learning_rate=0.05,
                                                              max_depth=5, subsample=0.8,
                                                              min_samples_leaf=20, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05,
                                                max_depth=5, subsample=0.8,
                                                colsample_bytree=0.8, n_jobs=-1,
                                                eval_metric='logloss',
                                                random_state=42, verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                  max_depth=5, num_leaves=31, subsample=0.8,
                                                  n_jobs=-1, random_state=42, verbose=-1)
    if not include_slow:
        models.pop('Gradient Boosting', None)
    return models


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return '█' * filled + '░' * (width - filled)


def train_regression(models: dict, X_train, X_test, y_train, y_test,
                     scaler: StandardScaler) -> list:
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0
        preds = model.predict(Xte)
        r2  = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        results.append({'name': name, 'r2': r2, 'mae': mae, 'rmse': rmse,
                        'time': elapsed, 'model': model})
        print(f'    {name:<22} R²={r2:>6.3f}  MAE={mae:>5.2f}  RMSE={rmse:>5.2f}  '
              f'[{elapsed:>5.1f}s]')
    results.sort(key=lambda x: x['r2'], reverse=True)
    return results


def train_classification(models: dict, X_train, X_test, y_train, y_test,
                          scaler: StandardScaler) -> list:
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0
        preds      = model.predict(Xte)
        proba      = model.predict_proba(Xte)[:, 1] if hasattr(model, 'predict_proba') else preds
        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, zero_division=0)
        auc  = roc_auc_score(y_test, proba)
        results.append({'name': name, 'accuracy': acc, 'f1': f1, 'auc': auc,
                        'time': elapsed, 'model': model})
        print(f'    {name:<22} Acc={acc:>6.1%}  F1={f1:>5.3f}  AUC={auc:>5.3f}  '
              f'[{elapsed:>5.1f}s]')
    results.sort(key=lambda x: x['auc'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Pretty print final tables
# ---------------------------------------------------------------------------

def print_regression_table(results: list) -> None:
    print()
    print('  ╔══════════════════════════════════════════════════════════════╗')
    print('  ║              REGRESSION — ranked by R²                      ║')
    print('  ╠══════════════════════════════════════════════════════════════╣')
    print(f"  ║  {'Model':<22} {'R²':>6}  {'MAE':>6}  {'RMSE':>6}  {'Bar (R²)':>20}  ║")
    print('  ╠══════════════════════════════════════════════════════════════╣')
    for i, r in enumerate(results):
        medal = ['🥇', '🥈', '🥉', '  ', '  ', '  ', '  '][min(i, 6)]
        bar   = _bar(max(r['r2'], 0))
        print(f"  ║ {medal} {r['name']:<22} {r['r2']:>6.3f}  {r['mae']:>6.2f}  {r['rmse']:>6.2f}  {bar}  ║")
    print('  ╚══════════════════════════════════════════════════════════════╝')
    print(f"  Current GBR baseline: R²=0.208  MAE=4.82  RMSE≈6.2")


def print_classification_table(results: list) -> None:
    print()
    print('  ╔══════════════════════════════════════════════════════════════╗')
    print('  ║         CLASSIFICATION (OVER/UNDER) — ranked by AUC         ║')
    print('  ╠══════════════════════════════════════════════════════════════╣')
    print(f"  ║  {'Model':<22} {'Acc':>7}  {'F1':>6}  {'AUC':>6}  {'Bar (AUC)':>20}  ║")
    print('  ╠══════════════════════════════════════════════════════════════╣')
    for i, r in enumerate(results):
        medal = ['🥇', '🥈', '🥉', '  ', '  ', '  ', '  '][min(i, 6)]
        bar   = _bar(r['auc'])
        print(f"  ║ {medal} {r['name']:<22} {r['accuracy']:>7.1%}  {r['f1']:>6.3f}  "
              f"{r['auc']:>6.3f}  {bar}  ║")
    print('  ╚══════════════════════════════════════════════════════════════╝')
    print(f"  Break-even accuracy at −110 odds: 52.4%")


# ---------------------------------------------------------------------------
# Save best models
# ---------------------------------------------------------------------------

def save_best(reg_results: list, clf_results: list,
              scaler_reg: StandardScaler, scaler_clf: StandardScaler,
              clf_feature_cols: list) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    best_reg = reg_results[0]
    reg_path = MODELS_DIR / 'best_regression_model.pkl'
    joblib.dump({
        'model':        best_reg['model'],
        'scaler':       scaler_reg,
        'feature_cols': FEATURE_COLS,
        'model_name':   best_reg['name'],
        'performance':  {'r2': best_reg['r2'], 'mae': best_reg['mae']},
    }, reg_path)
    print(f"\n  Saved best regression  ({best_reg['name']}) → {reg_path.name}")

    best_clf = clf_results[0]
    clf_path = MODELS_DIR / 'best_classifier_model.pkl'
    joblib.dump({
        'model':        best_clf['model'],
        'scaler':       scaler_clf,
        'feature_cols': clf_feature_cols,
        'model_name':   best_clf['name'],
        'performance':  {'accuracy': best_clf['accuracy'], 'auc': best_clf['auc']},
    }, clf_path)
    print(f"  Saved best classifier  ({best_clf['name']}) → {clf_path.name}")


def save_report(reg_results: list, clf_results: list) -> None:
    report = {
        'regression': [
            {'name': r['name'], 'r2': round(r['r2'], 4),
             'mae': round(r['mae'], 4), 'rmse': round(r['rmse'], 4),
             'train_time_s': round(r['time'], 1)}
            for r in reg_results
        ],
        'classification': [
            {'name': r['name'], 'accuracy': round(r['accuracy'], 4),
             'f1': round(r['f1'], 4), 'auc': round(r['auc'], 4),
             'train_time_s': round(r['time'], 1)}
            for r in clf_results
        ],
    }
    out = MODELS_DIR / 'comparison_report.json'
    MODELS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"  Full report → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare regression and classification models')
    parser.add_argument('--limit-matches', type=int, default=None,
                        help='Cap on match JSON files to load (default: all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction held out for evaluation (default: 0.2)')
    parser.add_argument('--use-db', action='store_true',
                        help='Load from valorant_matches.db instead of JSON files (faster)')
    parser.add_argument('--save-best', action='store_true',
                        help='Save the best regression and classification models to models/')
    parser.add_argument('--fast', action='store_true',
                        help='Skip slow sklearn GradientBoosting (XGBoost gives ~identical metrics)')
    args = parser.parse_args()

    print()
    print('════════════════════════════════════════════════════════════')
    print(' VALORANT KILL PREDICTION — MODEL COMPARISON')
    print('════════════════════════════════════════════════════════════')

    # ── Sync career stats DB before training ─────────────────────────────────
    print('\n[0/4] Syncing career stats DB...')
    try:
        from backfill_career_stats import load_match_data, compute_career_stats, \
            get_existing_career_names, backfill
        _bdf      = load_match_data()
        _bstats   = compute_career_stats(_bdf, min_maps=10)
        _bexist   = get_existing_career_names()
        _bnew     = backfill(_bstats, _bexist, dry_run=False)
        if _bnew:
            print(f'  Backfilled {_bnew} new players into vlr_players.db')
        else:
            print('  Career DB already up-to-date')
    except Exception as _be:
        print(f'  Warning: career DB sync failed ({_be}) — continuing with existing data')

    # ── Load data ────────────────────────────────────────────────────────────
    print('\n[1/4] Loading training data...')
    t0 = time.time()
    df = load_data(limit_matches=args.limit_matches, use_db=args.use_db)
    src = 'DB' if args.use_db else 'JSON files'
    print(f'  {len(df):,} rows loaded from {src} in {time.time()-t0:.1f}s')

    # ── Build feature sets ───────────────────────────────────────────────────
    print('\n[2/4] Building feature sets...')
    X_reg, y_reg, X_clf, y_clf, scaler_reg, scaler_clf = build_datasets(df)
    print(f'  Regression  : {X_reg.shape[1]} features')
    print(f'  Classification: {X_clf.shape[1]} features (adds {", ".join(CLF_EXTRA)})')

    # ── Chronological split (no leakage) ─────────────────────────────────────
    # Train on the oldest matches, evaluate on the most recent — the only honest
    # way to estimate live performance. A random split lets a player's future
    # maps inform the model about its own past, inflating every metric.
    if 'match_date' in df.columns:
        dates = pd.to_datetime(df['match_date'].values, errors='coerce')
        order = np.argsort(dates.values, kind='mergesort')  # stable, NaT sorts last
    else:
        print('  WARNING: no match_date column — falling back to random split')
        order = np.random.RandomState(42).permutation(len(X_reg))

    n_test    = int(len(order) * args.test_size)
    train_idx = order[:-n_test]
    test_idx  = order[-n_test:]

    X_reg_tr, X_reg_te = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_reg_tr, y_reg_te = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    X_clf_tr, X_clf_te = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
    y_clf_tr, y_clf_te = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    # Refit scalers on TRAIN ONLY (build_datasets fit them on all rows, which
    # leaks test-set feature distribution into the transform).
    scaler_reg = StandardScaler().fit(X_reg_tr)
    scaler_clf = StandardScaler().fit(X_clf_tr)

    if 'match_date' in df.columns:
        tr_dates = pd.to_datetime(df['match_date'].iloc[train_idx])
        te_dates = pd.to_datetime(df['match_date'].iloc[test_idx])
        print(f'  Train: {len(X_reg_tr):,}  [{tr_dates.min():%Y-%m-%d} → {tr_dates.max():%Y-%m-%d}]')
        print(f'  Test : {len(X_reg_te):,}  [{te_dates.min():%Y-%m-%d} → {te_dates.max():%Y-%m-%d}]')
    else:
        print(f'  Train: {len(X_reg_tr):,}  |  Test: {len(X_reg_te):,}')

    # ── Regression ───────────────────────────────────────────────────────────
    print('\n[3/4] Training regression models...')
    reg_results = train_regression(
        regression_models(include_slow=not args.fast),
        X_reg_tr, X_reg_te, y_reg_tr, y_reg_te, scaler_reg)

    # ── Classification ───────────────────────────────────────────────────────
    print('\n[4/4] Training classification models...')
    clf_feature_cols = FEATURE_COLS + CLF_EXTRA
    clf_results = train_classification(
        classification_models(include_slow=not args.fast),
        X_clf_tr, X_clf_te, y_clf_tr, y_clf_te, scaler_clf)

    # ── Results tables ───────────────────────────────────────────────────────
    print()
    print('════════════════════════════════════════════════════════════')
    print(' RESULTS')
    print('════════════════════════════════════════════════════════════')
    print_regression_table(reg_results)
    print_classification_table(clf_results)

    # ── Winner summary ───────────────────────────────────────────────────────
    best_reg = reg_results[0]
    best_clf = clf_results[0]
    print()
    print('  Best regression  :', best_reg['name'],
          f"R²={best_reg['r2']:.3f}  MAE={best_reg['mae']:.2f}")
    print('  Best classifier  :', best_clf['name'],
          f"Accuracy={best_clf['accuracy']:.1%}  AUC={best_clf['auc']:.3f}")
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    save_report(reg_results, clf_results)
    if args.save_best:
        save_best(reg_results, clf_results, scaler_reg, scaler_clf, clf_feature_cols)


if __name__ == '__main__':
    main()
