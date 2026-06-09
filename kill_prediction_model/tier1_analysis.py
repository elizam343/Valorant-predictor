"""Tier-1 batch analysis (backlog #2 L1 objective, #3 residual analysis, #4 ensemble).

Loads the training data ONCE, reproduces model_comparison's chronological split,
then:
  #2  trains XGB/LGBM with L2 (squared) vs L1 (absolute) objectives and compares MAE
  #4  evaluates a simple average ensemble of the two
  #3  takes the best model and reports holdout MAE bucketed by player / role /
      favored-vs-dog / blowout-vs-grind / form / actual-kill magnitude, plus the
      mean-bias check (is mean(pred) > mean(actual)?)

Run:  python tier1_analysis.py
"""
import numpy as np
import pandas as pd

from enhanced_data_loader import EnhancedDataLoader
from model_comparison import FEATURE_COLS

try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit(f'need xgboost + lightgbm: {e}')

TEST_SIZE = 0.2
HP = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    print('=== Loading data (this is the slow part) ===')
    X, y = EnhancedDataLoader().prepare_training_data()
    df = X.copy()
    df['match_kills'] = y.values

    # Chronological split — identical to model_comparison
    dates = pd.to_datetime(df['match_date'].values, errors='coerce')
    order = np.argsort(dates.values, kind='mergesort')
    n_test = int(len(order) * TEST_SIZE)
    tr, te = order[:-n_test], order[-n_test:]

    Xr = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    yr = df['match_kills']
    Xtr, Xte = Xr.iloc[tr], Xr.iloc[te]
    ytr, yte = yr.iloc[tr].values, yr.iloc[te].values
    dte = df.iloc[te].reset_index(drop=True)   # full rows for bucketing
    print(f'  train={len(tr)}  test={len(te)}  features={len(FEATURE_COLS)}')

    # ── #2 + #4: objective comparison + ensemble ──────────────────────────────
    print('\n=== #2 L1 vs L2 objective  |  #4 ensemble ===')
    fitted = {}
    specs = {
        'XGB L2':  xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, verbosity=0, **HP),
        'XGB L1':  xgb.XGBRegressor(objective='reg:absoluteerror', n_jobs=-1, verbosity=0, **HP),
        'LGBM L2': lgb.LGBMRegressor(objective='regression',     n_jobs=-1, verbose=-1, num_leaves=31, **HP),
        'LGBM L1': lgb.LGBMRegressor(objective='regression_l1',  n_jobs=-1, verbose=-1, num_leaves=31, **HP),
    }
    preds = {}
    for name, m in specs.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        fitted[name] = m
        print(f'  {name:9} MAE={mae(yte,p):.4f}  R2={1-np.sum((yte-p)**2)/np.sum((yte-yte.mean())**2):.4f}')

    ens_l1 = (preds['XGB L1'] + preds['LGBM L1']) / 2
    ens_l2 = (preds['XGB L2'] + preds['LGBM L2']) / 2
    print(f'  {"ENS L1":9} MAE={mae(yte,ens_l1):.4f}   (avg of XGB L1 + LGBM L1)')
    print(f'  {"ENS L2":9} MAE={mae(yte,ens_l2):.4f}   (avg of XGB L2 + LGBM L2)')

    # Pick best by MAE for residual analysis
    cands = {**preds, 'ENS L1': ens_l1, 'ENS L2': ens_l2}
    best = min(cands, key=lambda k: mae(yte, cands[k]))
    bp = cands[best]
    print(f'\n  >> best by MAE: {best} (MAE={mae(yte,bp):.4f})  — used for residual analysis')

    # ── #3: residual analysis on the best model ───────────────────────────────
    print('\n=== #3 Residual analysis (' + best + ') ===')
    resid = bp - yte                       # +ve = over-prediction
    print(f'  mean(pred)={bp.mean():.2f}  mean(actual)={yte.mean():.2f}  '
          f'BIAS={resid.mean():+.3f}  (if >0, recentering is a free MAE cut)')
    print(f'  overall test MAE={mae(yte,bp):.4f}')

    def bucket(label, keys):
        print(f'\n  -- MAE by {label} --')
        g = pd.DataFrame({'k': keys, 'ae': np.abs(resid), 'bias': resid})
        agg = g.groupby('k').agg(n=('ae','size'), MAE=('ae','mean'), bias=('bias','mean'))
        agg = agg[agg['n'] >= 30].sort_values('MAE', ascending=False)
        for k, r in agg.iterrows():
            print(f'     {str(k):26} n={int(r.n):5}  MAE={r.MAE:.3f}  bias={r.bias:+.3f}')

    role_map = {0:'Sentinel',1:'Controller',2:'Initiator',3:'Duelist',1.5:'Unknown'}
    bucket('role', dte['agent_role_ordinal'].map(role_map).fillna('Unknown'))

    fav = np.where(dte['team_strength'] - dte['opponent_team_strength'] > 0.02, 'favored',
          np.where(dte['team_strength'] - dte['opponent_team_strength'] < -0.02, 'underdog', 'even'))
    bucket('matchup (team_strength gap)', fav)

    rounds = dte['avg_rounds_vs_opponent']
    bucket('expected rounds (blowout<->grind)',
           pd.cut(rounds, [0,18,21,24,100], labels=['<18 blowout','18-21','21-24','24+ grind']))

    bucket('actual kills (magnitude)',
           pd.cut(yte, [0,12,17,22,100], labels=['<=12 low','13-17','18-22','23+ high']))

    bucket('form_slope sign',
           np.where(dte['form_slope']>0.05,'rising',np.where(dte['form_slope']<-0.05,'falling','flat')))

    # worst individual players (>=20 test maps)
    print('\n  -- worst players (>=20 test maps) --')
    g = pd.DataFrame({'p': dte.get('player_name', pd.Series(['?']*len(dte))),
                      'ae': np.abs(resid), 'bias': resid})
    pl = g.groupby('p').agg(n=('ae','size'), MAE=('ae','mean'), bias=('bias','mean'))
    pl = pl[pl['n'] >= 20].sort_values('MAE', ascending=False).head(10)
    for k, r in pl.iterrows():
        print(f'     {str(k):20} n={int(r.n):4}  MAE={r.MAE:.3f}  bias={r.bias:+.3f}')


if __name__ == '__main__':
    main()
