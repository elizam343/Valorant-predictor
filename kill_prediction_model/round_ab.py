"""Round-count Stage 2a — feature-swap A/B (#10, analysis #2).

On the scored corpus (true rounds), train the kill model two ways differing ONLY
in how `avg_rounds_vs_opponent` is computed:
  A. total_kills / 6  (the proxy — production default)
  B. TRUE rounds (team1+team2 from the scored scoreboard)
Same chronological split, same 24 features otherwise, XGBoost-L1. Reports ΔMAE
(averaged over seeds). Leak-safe: avg_rounds_vs_opponent is a leave-one-out
historical average per team-matchup, never this map's own rounds.

Run: python round_ab.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import xgboost as xgb

from enhanced_data_loader import EnhancedDataLoader
from features import FEATURE_COLS

SCORED_DIR = os.path.join(os.path.dirname(__file__), '..', 'scraped_matches_scored')
HP = dict(objective='reg:absoluteerror', n_estimators=400, learning_rate=0.05,
          max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0)


def true_rounds_lookup():
    """ {(match_id_str, map_name): true_rounds} from the scored JSONs. """
    lut = {}
    for fp in glob.glob(os.path.join(SCORED_DIR, 'match_*.json')):
        mid = os.path.basename(fp).replace('match_', '').replace('.json', '')
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for m in d.get('map_stats') or []:
            ts = m.get('total_score')
            if ts:
                lut[(mid, m.get('map') or m.get('map_name', 'Unknown'))] = ts.get('team1', 0) + ts.get('team2', 0)
    return lut


def build_frame():
    ld = EnhancedDataLoader(scraped_matches_dir=SCORED_DIR)
    matches = ld.load_scraped_matches()
    df = ld.create_training_dataset(matches)
    df = ld.create_kill_prediction_features(df)
    df = ld.add_rolling_features(df)
    df = ld.add_agent_features(df)

    # True rounds per (match, map) → leave-one-out avg by team-matchup (mirror of
    # the proxy at enhanced_data_loader.py:697-702, but on real rounds).
    lut = true_rounds_lookup()
    df['true_rounds'] = [lut.get((str(mi), mn), np.nan)
                         for mi, mn in zip(df['match_id'], df['map_name'])]
    df = df[df['true_rounds'].notna()].copy()
    grp = df.groupby(['team', 'opponent_team'])['true_rounds']
    s, c = grp.transform('sum'), grp.transform('count')
    df['avg_rounds_true'] = ((s - df['true_rounds']) / (c - 1)).fillna(df['true_rounds'].mean())

    df = df[(df['db_rating'] > 0) & (df['match_kills'] > 0)].copy()
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    df = build_frame()
    n = len(df)
    n_test = int(n * 0.2)
    tr, te = slice(0, n - n_test), slice(n - n_test, n)
    y = df['match_kills'].values

    print(f'Scored A/B set: {n} player-maps ({df["match_id"].nunique()} matches), '
          f'train {n-n_test} / test {n_test}')
    rp = np.corrcoef(df['avg_rounds_vs_opponent'].fillna(df['avg_rounds_vs_opponent'].mean()),
                     df['avg_rounds_true'])[0, 1]
    print(f'corr(proxy avg_rounds, true avg_rounds) = {rp:.3f}\n')

    # Version A = production features (proxy). Version B = swap the one column.
    Xa = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    cols_b = [c for c in FEATURE_COLS]
    Xb = Xa.copy()
    Xb['avg_rounds_vs_opponent'] = df['avg_rounds_true'].values   # the only change

    maes_a, maes_b = [], []
    for seed in range(5):
        hp = dict(HP, random_state=seed)
        ma = xgb.XGBRegressor(**hp).fit(Xa.iloc[tr], y[tr]); pa = ma.predict(Xa.iloc[te])
        mb = xgb.XGBRegressor(**hp).fit(Xb.iloc[tr], y[tr]); pb = mb.predict(Xb.iloc[te])
        maes_a.append(mae(y[te], pa)); maes_b.append(mae(y[te], pb))

    a, b = np.mean(maes_a), np.mean(maes_b)
    print(f'{"":18}{"MAE":>8}{"±sd":>8}')
    print(f'{"A proxy (kills/6)":18}{a:>8.4f}{np.std(maes_a):>8.4f}')
    print(f'{"B true rounds":18}{b:>8.4f}{np.std(maes_b):>8.4f}')
    print(f'\nΔMAE (proxy − true) = {a-b:+.4f}   ({(a-b)/a*100:+.2f}%)')
    if b < a - 0.01:
        print('>> true rounds HELP — Stage 2b reformulation likely worth building.')
    elif b > a + 0.01:
        print('>> true rounds HURT here — proxy carries useful kill info; skip 2b feature-swap.')
    else:
        print('>> FLAT (|Δ| < 0.01) — round feature swap is a wash; round-count not worth more here.')


if __name__ == '__main__':
    main()
