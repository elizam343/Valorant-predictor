"""Player random-effect — Stage A offline test (#12).

Adds a CAUSAL, shrunk per-player baseline (mean kills over strictly-earlier maps,
empirical-Bayes shrunk toward the global mean) to the 24 features and measures
ΔMAE on the chronological holdout. Leak-safe: prior maps only — never this map.
No production changes; this is the go/no-go for integrating it.

Run: python player_effect_test.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from enhanced_data_loader import EnhancedDataLoader
from features import FEATURE_COLS

K_SHRINK = 10        # maps of shrinkage toward the global mean (low-sample guard)
HP = dict(objective='reg:absoluteerror', n_estimators=400, learning_rate=0.05,
          max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    X, y = EnhancedDataLoader().prepare_training_data()
    df = X.copy()
    df['match_kills'] = y.values
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

    # --- causal shrunk player baseline (prior maps only) ---
    df = df.sort_values(['player_name', 'match_date']).reset_index(drop=True)
    gm = df['match_kills'].mean()
    prior_sum = df.groupby('player_name')['match_kills'].cumsum() - df['match_kills']
    prior_cnt = df.groupby('player_name').cumcount()
    df['player_baseline'] = (prior_sum + K_SHRINK * gm) / (prior_cnt + K_SHRINK)
    print(f'global mean kills = {gm:.2f}; player_baseline range '
          f'[{df["player_baseline"].min():.1f}, {df["player_baseline"].max():.1f}]')
    print(f'corr(player_baseline, recent_avg_kills) = '
          f'{df["player_baseline"].corr(df["recent_avg_kills"]):.3f}  '
          f'(high → redundant with existing feature)')

    # --- chronological split ---
    df = df.sort_values('match_date', kind='mergesort').reset_index(drop=True)
    n_test = int(len(df) * 0.2)
    tr, te = slice(0, len(df) - n_test), slice(len(df) - n_test, len(df))
    yv = df['match_kills'].values

    Xa = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())               # baseline 24
    Xb = Xa.copy(); Xb['player_baseline'] = df['player_baseline'].values  # + the effect

    print(f'\ntrain {len(df)-n_test} / test {n_test}\n')
    A, B = [], []
    imp = []
    for seed in range(5):
        hp = dict(HP, random_state=seed)
        ma = xgb.XGBRegressor(**hp).fit(Xa.iloc[tr], yv[tr]); A.append(mae(yv[te], ma.predict(Xa.iloc[te])))
        mb = xgb.XGBRegressor(**hp).fit(Xb.iloc[tr], yv[tr]); B.append(mae(yv[te], mb.predict(Xb.iloc[te])))
        fi = dict(zip(Xb.columns, mb.feature_importances_))
        imp.append(fi['player_baseline'])

    a, b = np.mean(A), np.mean(B)
    print(f'{"":22}{"MAE":>8}{"±sd":>8}')
    print(f'{"A: 24 features":22}{a:>8.4f}{np.std(A):>8.4f}')
    print(f'{"B: + player_baseline":22}{b:>8.4f}{np.std(B):>8.4f}')
    print(f'\nΔMAE (base − +baseline) = {a-b:+.4f}  ({(a-b)/a*100:+.2f}%)')
    # where does the new feature rank in importance?
    ranks = sorted(dict(zip(Xb.columns, mb.feature_importances_)).items(), key=lambda x: -x[1])
    pos = [i for i, (c, _) in enumerate(ranks, 1) if c == 'player_baseline'][0]
    print(f'player_baseline importance rank: {pos}/{len(ranks)} (avg gain {np.mean(imp):.0f})')
    if b < a - 0.01:
        print('>> HELPS — integrate into production (Stage B).')
    elif b > a + 0.01:
        print('>> HURTS — skip.')
    else:
        print('>> FLAT (|Δ|<0.01) — redundant with existing features; not worth integrating.')


if __name__ == '__main__':
    main()
