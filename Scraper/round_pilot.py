"""Round-count Stage 1a pilot (#9): re-scrape recent VLR matches with the fixed
parser, then quantify how wrong the `total_kills / 6` rounds proxy is vs TRUE
rounds (team1_score + team2_score). This number is the go/no-go for Stages 1b/2.

Self-contained: scrapes a bounded match-ID window, saves scored JSON to
scraped_matches_scored/ (does NOT touch the 52k corpus), and writes
round_proxy_validation.csv + a printed summary. Rate-limited + resumable.

Usage:  python round_pilot.py [START_ID] [END_ID] [MAX_VALID]
"""
import csv, json, os, sys, time
import numpy as np
import scraper_api

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'scraped_matches_scored')
REPORT  = os.path.join(os.path.dirname(__file__), 'round_proxy_validation.csv')
PROXY_DIVISOR = 6.0          # the divisor the current avg_rounds proxy uses
SLEEP = 0.6


def kills_sum(map_stat):
    tot = n = 0
    for p in map_stat.get('flat_players', []):
        try:
            tot += int(str(p.get('kills', '')).strip()); n += 1
        except (ValueError, TypeError):
            pass
    return tot, n


def collect(start, end, max_valid, rows, seen):
    """Scan an ID window, append (mid, map, tk, proxy, true) to rows (deduped)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    valid = scanned = 0
    for mid in range(start, end):
        if valid >= max_valid:
            break
        scanned += 1
        out = os.path.join(OUT_DIR, f'match_{mid}.json')
        try:
            if os.path.exists(out):
                d = json.load(open(out))             # resume: reuse already-scraped
            else:
                d = scraper_api.scrape_match_details(mid)
                time.sleep(SLEEP)
        except Exception:
            time.sleep(SLEEP); continue

        got = False
        for m in d.get('map_stats') or []:
            ts = m.get('total_score')
            if not ts:
                continue
            tr = ts.get('team1', 0) + ts.get('team2', 0)
            if tr < 13 or tr > 40:                   # sane completed map only
                continue
            key = (mid, m.get('map', ''))
            if key in seen:
                continue
            tk, npl = kills_sum(m)
            if npl < 8 or tk <= 0:                   # need ~full 10-player map
                continue
            seen.add(key)
            rows.append((mid, m.get('map', ''), tk, tk / PROXY_DIVISOR, tr))
            got = True
        if got:
            valid += 1
            json.dump(d, open(out, 'w'))
    return valid


def _stats(rows):
    proxy = np.array([r[3] for r in rows]); true = np.array([r[4] for r in rows])
    tk = np.array([r[2] for r in rows])
    r = float(np.corrcoef(proxy, true)[0, 1]) if len(rows) > 2 else float('nan')
    return dict(n=len(rows), r=r, mae=float(np.mean(np.abs(proxy - true))),
                bias=float(np.mean(proxy - true)), div=float(np.mean(tk / true)),
                tmean=float(true.mean()), tmin=float(true.min()), tmax=float(true.max()))


def report(rows, per_window=None):
    if not rows:
        print('No valid maps collected — widen the windows.'); return
    with open(REPORT, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['match_id', 'map', 'total_kills', 'proxy_rounds', 'true_rounds'])
        w.writerows(rows)
    if per_window:
        print('\n  -- per-window correlation (generalization check) --')
        for label, wr in per_window:
            if len(wr) > 2:
                s = _stats(wr)
                print(f'     {label:14} n={s["n"]:4}  r={s["r"]:.3f}  MAE={s["mae"]:.2f}  div={s["div"]:.2f}')
    s = _stats(rows)
    print('\n' + '=' * 56)
    print(f'  ROUND PROXY VALIDATION (combined)  —  {s["n"]} maps')
    print('=' * 56)
    print(f'  proxy = total_kills / {PROXY_DIVISOR}   vs   true = team1+team2 rounds')
    print(f'  correlation(proxy, true) : {s["r"]:.3f}   (r^2 = {s["r"]**2:.2f})')
    print(f'  proxy MAE (rounds)       : {s["mae"]:.2f}')
    print(f'  proxy bias (rounds)      : {s["bias"]:+.2f}')
    print(f'  empirical divisor        : {s["div"]:.2f}  (assumed {PROXY_DIVISOR})')
    print(f'  true rounds: mean {s["tmean"]:.1f}  range [{s["tmin"]:.0f}, {s["tmax"]:.0f}]')
    print(f'\n  report → {REPORT}')
    if s['r'] >= 0.9 and abs(s['bias']) < 1.0:
        print('  >> proxy is strong — Stage 1b/2 upside is LIMITED.')
    else:
        print('  >> proxy is weak/biased — real rounds (Stage 1b/2) likely WORTH it.')


# Event windows across the 2026 season (start, end, label) — sampled for generalization.
WINDOWS = [
    (660300, 660800, 'EMEA-S1'),
    (668000, 668600, 'mid-Jun-A'),
    (675000, 675600, 'mid-Jun-B'),
    (681200, 681800, 'EWC-AMER'),
    (684500, 685000, 'Masters-LDN'),
    (692200, 692800, 'playoffs'),
]


def run(start, end, max_valid=200):           # single-window (kept for compatibility)
    rows, seen = [], set()
    collect(start, end, max_valid, rows, seen)
    report(rows)


def run_multi(per_window=40):
    rows, seen, pw = [], set(), []
    for start, end, label in WINDOWS:
        before = len(rows)
        v = collect(start, end, per_window, rows, seen)
        pw.append((label, rows[before:]))
        print(f'  [{label}] {v} matches, {len(rows)-before} maps (total {len(rows)})', flush=True)
    report(rows, per_window=pw)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'multi':
        per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        print(f'Multi-event pilot: {len(WINDOWS)} windows, up to {per} matches each, sleep {SLEEP}s')
        run_multi(per)
    else:
        start = int(sys.argv[1]) if len(sys.argv) > 1 else 684000
        end   = int(sys.argv[2]) if len(sys.argv) > 2 else 684800
        mx    = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        print(f'Pilot scrape: IDs {start}..{end}, up to {mx} valid matches, sleep {SLEEP}s')
        run(start, end, mx)
