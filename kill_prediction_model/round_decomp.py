"""Round-count Stage 2a — decomposition analysis (#10, analysis #1).

Uses the true-rounds corpus (scraped_matches_scored/) to SIZE the round-count
prize before any retrain:

  A. kills ~ rounds      — how much of per-map kill variance do rounds explain?
                           (pooled, and the more honest WITHIN-player version)
  B. KPR stability       — is a player's KPR (kills/round) more stable across maps
                           than their raw kills? If yes → predicting KPR × rounds
                           beats predicting kills directly (the Stage 2b thesis).
  C. rounds ~ mismatch   — does pre-match team-strength gap predict round count
                           (blowouts)? Bounds how PREDICTABLE rounds are pre-match.

No model, no network — just the scored JSONs. Run: python round_decomp.py
"""
import glob, json, os
import numpy as np
import pandas as pd

SCORED_DIR = os.path.join(os.path.dirname(__file__), '..', 'scraped_matches_scored')
MIN_MAPS_PER_PLAYER = 15      # for the within-player / KPR-stability analyses


def load_rows():
    rows = []
    for fp in glob.glob(os.path.join(SCORED_DIR, 'match_*.json')):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for m in d.get('map_stats') or []:
            ts = m.get('total_score')
            if not ts:
                continue
            rounds = ts.get('team1', 0) + ts.get('team2', 0)
            if rounds < 13 or rounds > 40:
                continue
            fps = m.get('flat_players') or []
            # team average rating (pre-match strength proxy)
            team_rat = {}
            for p in fps:
                try:
                    team_rat.setdefault(p.get('team', ''), []).append(float(p.get('rating', 0)))
                except (ValueError, TypeError):
                    pass
            team_avg = {t: np.mean(v) for t, v in team_rat.items() if v}
            if len(team_avg) != 2:
                continue
            for p in fps:
                try:
                    k = int(str(p.get('kills', '')).strip())
                except (ValueError, TypeError):
                    continue
                if k <= 0:
                    continue
                team = p.get('team', '')
                opp = [t for t in team_avg if t != team]
                gap = (team_avg.get(team, np.nan) - team_avg[opp[0]]) if opp else np.nan
                rows.append((d.get('match_id'), m.get('map', ''), p.get('name', '').lower(),
                             team, k, rounds, k / rounds, gap))
    return pd.DataFrame(rows, columns=['match', 'map', 'player', 'team',
                                       'kills', 'rounds', 'kpr', 'gap'])


def cv(x):
    x = np.asarray(x, float)
    return x.std() / x.mean() if x.mean() else np.nan


def main():
    df = load_rows()
    print(f'Loaded {len(df)} player-map rows from {df["match"].nunique()} matches '
          f'({df["player"].nunique()} players)\n')

    # ---- A. kills ~ rounds ----
    r_pool = np.corrcoef(df['rounds'], df['kills'])[0, 1]
    print('== A. kills ~ rounds ==')
    print(f'  pooled correlation r = {r_pool:.3f}   (r^2 = {r_pool**2:.3f})')
    # within-player: avg R^2 of kills~rounds across players with enough maps
    wr = []
    for p, g in df.groupby('player'):
        if len(g) >= MIN_MAPS_PER_PLAYER and g['rounds'].std() > 0:
            rr = np.corrcoef(g['rounds'], g['kills'])[0, 1]
            if np.isfinite(rr):
                wr.append(rr ** 2)
    print(f'  within-player avg r^2 = {np.mean(wr):.3f}  (n={len(wr)} players ≥{MIN_MAPS_PER_PLAYER} maps)')
    print(f'  → rounds explain ~{np.mean(wr)*100:.0f}% of a player\'s game-to-game kill variance\n')

    # ---- B. KPR stability vs raw-kills stability ----
    print('== B. KPR stability (Stage 2b thesis) ==')
    kills_cv, kpr_cv, better = [], [], 0
    for p, g in df.groupby('player'):
        if len(g) < MIN_MAPS_PER_PLAYER:
            continue
        ck, cp = cv(g['kills']), cv(g['kpr'])
        if np.isfinite(ck) and np.isfinite(cp):
            kills_cv.append(ck); kpr_cv.append(cp); better += (cp < ck)
    n = len(kills_cv)
    print(f'  mean CV(kills) = {np.mean(kills_cv):.3f}   mean CV(KPR) = {np.mean(kpr_cv):.3f}')
    print(f'  KPR is MORE stable for {better}/{n} players ({better/n*100:.0f}%)')
    verdict = 'YES → KPR×rounds should beat predicting kills' if np.mean(kpr_cv) < np.mean(kills_cv) \
        else 'NO → KPR not more stable; reformulation upside limited'
    print(f'  → {verdict}\n')

    # ---- C. rounds ~ pre-match strength mismatch ----
    print('== C. rounds ~ team-strength mismatch ==')
    sub = df.dropna(subset=['gap']).copy()
    sub['absgap'] = sub['gap'].abs()
    # one row per (match, map): total rounds vs the matchup's strength gap
    permap = sub.groupby(['match', 'map']).agg(rounds=('rounds', 'first'),
                                               absgap=('absgap', 'max')).reset_index()
    rc = np.corrcoef(permap['absgap'], permap['rounds'])[0, 1]
    print(f'  corr(|strength gap|, rounds) = {rc:.3f}  (negative = mismatch → fewer rounds/blowouts)')
    q = pd.qcut(permap['absgap'], 4, labels=['even', 'q2', 'q3', 'lopsided'])
    print('  mean rounds by mismatch quartile:')
    for lab, gg in permap.groupby(q, observed=True):
        print(f'     {str(lab):9} n={len(gg):4}  mean rounds={gg["rounds"].mean():.1f}')
    print(f'\n  → rounds are {"somewhat" if abs(rc)>0.15 else "barely"} predictable from pre-match strength '
          f'(bounds Stage 2b\'s expected-rounds model).')


if __name__ == '__main__':
    main()
