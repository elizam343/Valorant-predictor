#!/usr/bin/env python3
"""
Backtester / performance analytics for the Valorant kill-line model.

Turns the logged history into the metrics that actually matter for betting —
ROI at -110, hit rate by edge band, classifier accuracy on REAL lines, and
probability calibration — rather than R²/AUC on a held-out split.

Two data sources (both produced by the daily pipeline):
  - bet_results.csv   : the bets we actually recommended (settled W/L/PUSH)
  - line_history.csv  : the FULL market — every line seen, bet or not, with the
                        realized OVER/UNDER label once results are entered

Usage
-----
  python backtest.py                 # full report
  python backtest.py --since 2026-06-01
  python backtest.py --min-edge 15   # only strategy rows with edge >= 15

As the corpus grows this becomes the source of truth for "is the edge real?".
CLV (closing-line value) is stubbed — it needs lines logged again near match
start; see _clv_note().
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

BET_FILE  = Path(__file__).parent / 'bet_results.csv'
LINE_FILE = Path(__file__).parent / 'line_history.csv'

BREAK_EVEN = 0.524          # win rate needed at -110 to break even
WIN_PROFIT = 100.0          # units won on a 110-unit risk at -110
LOSS_COST  = 110.0


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _roi(wins: int, losses: int) -> tuple:
    """Return (win_rate_pct, pnl_units, roi_pct) at -110."""
    decided = wins + losses
    if decided == 0:
        return 0.0, 0.0, 0.0
    pnl  = wins * WIN_PROFIT - losses * LOSS_COST
    risk = decided * LOSS_COST
    return wins / decided * 100, pnl, pnl / risk * 100


# ---------------------------------------------------------------------------
# 1) Realized bet performance (from bet_results.csv)
# ---------------------------------------------------------------------------

def report_bets(rows: List[Dict], since: str = None) -> None:
    settled = [r for r in rows
               if r.get('outcome') in ('WIN', 'LOSS', 'PUSH')
               and (not since or r['date'] >= since)]
    pending = [r for r in rows if r.get('outcome') == 'PENDING']

    print('\n  ── Realized bets (bet_results.csv) ─────────────────────────')
    if not settled:
        print(f'   No settled bets yet.  ({len(pending)} pending)')
        return

    wins   = sum(1 for r in settled if r['outcome'] == 'WIN')
    losses = sum(1 for r in settled if r['outcome'] == 'LOSS')
    pushes = sum(1 for r in settled if r['outcome'] == 'PUSH')
    wr, pnl, roi = _roi(wins, losses)
    sign = '+' if pnl >= 0 else ''
    print(f'   Settled : {len(settled)}   {wins}W / {losses}L / {pushes}P   ({len(pending)} pending)')
    print(f'   Win rate: {wr:.1f}%   (break-even {BREAK_EVEN*100:.1f}%)')
    print(f'   P&L     : {sign}{pnl:.0f}u   ROI: {sign}{roi:.1f}%   ($1/unit, -110)')

    bands = [('10–20%', 10, 20), ('20–30%', 20, 30),
             ('30–40%', 30, 40), ('40%+', 40, 1e9)]
    printed = False
    for label, lo, hi in bands:
        grp = [r for r in settled if lo <= _fnum(r.get('edge_pct'), 0) < hi]
        w = sum(1 for r in grp if r['outcome'] == 'WIN')
        l = sum(1 for r in grp if r['outcome'] == 'LOSS')
        if w + l == 0:
            continue
        if not printed:
            print('   By edge band:'); printed = True
        bwr, bpnl, broi = _roi(w, l)
        bsign = '+' if bpnl >= 0 else ''
        print(f'     {label:<7} {w}W/{l}L  {bwr:.0f}%  ROI {bsign}{broi:.0f}%')


# ---------------------------------------------------------------------------
# 2) Market-corpus analytics (from line_history.csv)
# ---------------------------------------------------------------------------

def report_corpus(rows: List[Dict], since: str = None, min_edge: float = 0.0) -> None:
    settled = [r for r in rows
               if r.get('line_result') in ('OVER', 'UNDER', 'PUSH')
               and (not since or r['date'] >= since)]

    print('\n  ── Market corpus (line_history.csv) ────────────────────────')
    print(f'   Logged lines : {len(rows)}   settled: {len(settled)}')
    if not settled:
        print('   No settled lines yet — populates as you enter results daily.')
        print('   Run:  python results_tracker.py result <player> <actual>')
        return

    # Classifier accuracy on REAL lines (the metric step 4 aims to improve)
    graded = [r for r in settled
              if r.get('line_result') != 'PUSH' and _fnum(r.get('prob_over')) is not None]
    if graded:
        correct = sum(
            1 for r in graded
            if ('OVER' if _fnum(r['prob_over']) >= 0.5 else 'UNDER') == r['line_result']
        )
        print(f'   Classifier acc on real lines: {correct}/{len(graded)} '
              f'= {correct/len(graded)*100:.1f}%')

    # Strategy replay: bet our `rec` on actual recommendations
    strat = [r for r in settled
             if r.get('bet_rec') == 'BET'
             and r.get('line_result') != 'PUSH'
             and _fnum(r.get('edge_pct'), 0) >= min_edge]
    if strat:
        w = sum(1 for r in strat if r['rec'] == r['line_result'])
        l = len(strat) - w
        swr, spnl, sroi = _roi(w, l)
        sign = '+' if spnl >= 0 else ''
        print(f'   Strategy (BET, edge>={min_edge:g}): {w}W/{l}L  {swr:.1f}%  '
              f'ROI {sign}{sroi:.1f}%')

    _calibration(graded)
    _clv_note()


def _calibration(graded: List[Dict]) -> None:
    """Reliability: predicted P(OVER) vs realized OVER rate, in buckets."""
    if len(graded) < 10:
        return
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]
    print('   Calibration (P(OVER) → actual OVER rate):')
    for lo, hi in buckets:
        grp = [r for r in graded if lo <= _fnum(r['prob_over']) < hi]
        if not grp:
            continue
        actual_over = sum(1 for r in grp if r['line_result'] == 'OVER') / len(grp)
        pred_mean = sum(_fnum(r['prob_over']) for r in grp) / len(grp)
        print(f'     pred~{pred_mean:.2f}  actual {actual_over:.2f}  (n={len(grp)})')


def _clv_note() -> None:
    print('   CLV: not yet available — log lines again near match start to '
          'capture closing-line value.')


def main() -> None:
    ap = argparse.ArgumentParser(description='Backtest / performance analytics')
    ap.add_argument('--since', default=None, help='Only rows on/after YYYY-MM-DD')
    ap.add_argument('--min-edge', type=float, default=0.0,
                    help='Min edge for the strategy replay (default: 0)')
    args = ap.parse_args()

    print('\n  ════════════════════════════════════════════════════════════')
    print('   VALORANT KILL-LINE BACKTEST')
    print('  ════════════════════════════════════════════════════════════')
    report_bets(_load(BET_FILE), since=args.since)
    report_corpus(_load(LINE_FILE), since=args.since, min_edge=args.min_edge)
    print()


if __name__ == '__main__':
    main()
