#!/usr/bin/env python3
"""
Results tracker for daily kill line slate picks.

Saves picks to bet_results.csv, accepts actual outcomes, and
shows running performance stats at −110 odds.

Usage
-----
  python results_tracker.py pending
      Show all picks waiting for a result.

  python results_tracker.py result blowz 31
      blowz actually hit 31 kills M1+M2.

  python results_tracker.py result xavi8k 28 --date 2026-06-06
      Enter result for a specific past date.

  python results_tracker.py stats
      Win rate, P&L, and accuracy by edge band.

  python results_tracker.py history
  python results_tracker.py history --n 30
      Show full (or last N) result log.
"""

import argparse
import csv
import os
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_FILE = Path(__file__).parent / 'bet_results.csv'

COLUMNS = [
    'date', 'player', 'line', 'pred_total', 'edge_pct', 'rec',
    'team', 'opponent', 'maps', 'agent', 'maps_guessed', 'agent_guessed',
    'actual', 'outcome',
]

# Full daily market corpus — every PrizePicks line we saw, bet or not. This is
# the foundation for backtesting (ROI/CLV) and for training the model on REAL
# lines instead of synthetic ones. `line_result` is the market label
# (did actual kills go OVER/UNDER the line), independent of our recommendation.
LINE_HISTORY_FILE = Path(__file__).parent / 'line_history.csv'

LINE_HISTORY_COLUMNS = [
    'date', 'player', 'pp_line', 'pred_per_map', 'pred_total', 'prob_over',
    'p_win', 'edge_pct', 'kelly_stake', 'rec', 'bet_rec', 'hit_rate', 'hist_maps',
    'hist_source', 'team', 'opponent', 'agent', 'filter_reason', 'actual', 'line_result',
]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _save_file(path: Path, rows: List[Dict], columns: List[str]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _load() -> List[Dict]:
    return _load_file(RESULTS_FILE)


def _save(rows: List[Dict]) -> None:
    _save_file(RESULTS_FILE, rows, COLUMNS)


def _outcome(rec: str, line: float, actual: float) -> str:
    if actual > line:
        return 'WIN' if rec == 'OVER' else 'LOSS'
    if actual < line:
        return 'WIN' if rec == 'UNDER' else 'LOSS'
    return 'PUSH'


# ---------------------------------------------------------------------------
# Public API (called from bet_slate.py)
# ---------------------------------------------------------------------------

def save_slate(rows: List[Dict], today: Optional[str] = None) -> int:
    """
    Append bet rows to bet_results.csv.
    Skips any row whose (date, player) pair already exists.
    Returns the number of new rows written.

    Expects the dict format produced by bet_slate.build_slate():
      player, line, pred_total, edge, rec, context{team,opponent,maps,agent},
      maps_guessed, agent_guessed
    """
    today = today or str(_date.today())
    existing = _load()
    existing_keys = {(r['date'], r['player'].lower()) for r in existing}

    added = 0
    for r in rows:
        key = (today, r['player'].lower())
        if key in existing_keys:
            continue
        existing_keys.add(key)

        ctx = r.get('context', {})
        maps_list = ctx.get('maps') or []
        maps_str  = '/'.join(maps_list) if isinstance(maps_list, list) else str(maps_list)

        existing.append({
            'date':          today,
            'player':        r['player'],
            'line':          f"{float(r['line']):.1f}",
            'pred_total':    f"{float(r['pred_total']):.1f}",
            'edge_pct':      f"{float(r['edge']):.1f}",
            'rec':           r['rec'],
            'team':          ctx.get('team', ''),
            'opponent':      ctx.get('opponent', ''),
            'maps':          maps_str,
            'agent':         ctx.get('agent', ''),
            'maps_guessed':  str(r.get('maps_guessed', False)),
            'agent_guessed': str(r.get('agent_guessed', False)),
            'actual':        '',
            'outcome':       'PENDING',
        })
        added += 1

    _save(existing)
    return added


def log_line_history(rows: List[Dict], today: Optional[str] = None) -> int:
    """
    Append the FULL slate (every fetched line — bets, NO-BETs, all) to
    line_history.csv. Dedupes on (date, player). Returns rows added.

    This is the market corpus: over time it accumulates real PrizePicks lines
    paired with realized kills, which is what the backtester and a market-aware
    model are trained/evaluated on. Expects bet_slate.build_slate() row dicts.
    """
    today = today or str(_date.today())
    existing = _load_file(LINE_HISTORY_FILE)
    existing_keys = {(r['date'], r['player'].lower()) for r in existing}

    def _fmt(v, nd=1):
        return f'{float(v):.{nd}f}' if v is not None and v != '' else ''

    added = 0
    for r in rows:
        key = (today, str(r['player']).lower())
        if key in existing_keys:
            continue
        existing_keys.add(key)

        ctx = r.get('context', {})
        existing.append({
            'date':          today,
            'player':        r['player'],
            'pp_line':       _fmt(r.get('line')),
            'pred_per_map':  _fmt(r.get('pred_per_map')),
            'pred_total':    _fmt(r.get('pred_total')),
            'prob_over':     _fmt(r.get('prob_over'), 3),
            'p_win':         _fmt(r.get('p_win'), 3),
            'edge_pct':      _fmt(r.get('edge')),
            'kelly_stake':   _fmt(r.get('stake'), 3),
            'rec':           r.get('rec', ''),
            'bet_rec':       r.get('bet_rec', ''),
            'hit_rate':      _fmt(r.get('hit_rate'), 3),
            'hist_maps':     str(r.get('n_maps', '')),
            'hist_source':   r.get('hist_source', ''),
            'team':          ctx.get('team', ''),
            'opponent':      ctx.get('opponent', ''),
            'agent':         ctx.get('agent', ''),
            'filter_reason': r.get('filter_reason', ''),
            'actual':        '',
            'line_result':   'PENDING',
        })
        added += 1

    _save_file(LINE_HISTORY_FILE, existing, LINE_HISTORY_COLUMNS)
    return added


def _settle_line_history(player_lower: str, actual: float,
                         date: Optional[str] = None) -> bool:
    """Fill the realized kills + market label into line_history.csv for the
    most recent pending row matching this player (optionally a given date)."""
    rows = _load_file(LINE_HISTORY_FILE)
    cands = [
        (i, r) for i, r in enumerate(rows)
        if r['player'].lower() == player_lower
        and r.get('line_result', 'PENDING') == 'PENDING'
        and (not date or r['date'] == date)
    ]
    if not cands and date:
        cands = [(i, r) for i, r in enumerate(rows)
                 if r['player'].lower() == player_lower
                 and r.get('line_result', 'PENDING') == 'PENDING']
    if not cands:
        return False

    idx, row = max(cands, key=lambda x: x[1]['date'])
    line = float(row['pp_line']) if row.get('pp_line') else 0.0
    rows[idx]['actual'] = f'{actual:.1f}'
    rows[idx]['line_result'] = (
        'OVER' if actual > line else 'UNDER' if actual < line else 'PUSH'
    )
    _save_file(LINE_HISTORY_FILE, rows, LINE_HISTORY_COLUMNS)
    return True


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_pending(args) -> None:
    rows = [r for r in _load() if r['outcome'] == 'PENDING']
    if not rows:
        print('\n  No pending picks.\n')
        return

    print(f'\n  Pending picks  ({len(rows)})\n')
    print(f"  {'Date':<12} {'Player':<20} {'Line':>8} {'Pred':>8} {'Edge':>7}  {'Rec':<6}")
    print('  ' + '-' * 66)
    for r in rows:
        print(f"  {r['date']:<12} {r['player']:<20} "
              f"{float(r['line']):>8.1f} {float(r['pred_total']):>8.1f} "
              f"{float(r['edge_pct']):>6.1f}%  {r['rec']:<6}")
    print()


def cmd_result(args) -> None:
    rows = _load()
    player_lower = args.player.strip().lower()

    # Try date-specific first, then any pending for that player
    def _matches(r, require_date=False):
        return (
            r['player'].lower() == player_lower
            and r['outcome'] == 'PENDING'
            and (not require_date or r['date'] == args.date)
        )

    candidates = [
        (i, r) for i, r in enumerate(rows)
        if _matches(r, require_date=bool(args.date))
    ]
    if not candidates and args.date:
        candidates = [(i, r) for i, r in enumerate(rows) if _matches(r)]

    if not candidates:
        # Not a pending BET — but it may still be a logged MARKET line (bet or
        # not). Settle that so the slate-wide corpus is complete (needed for the
        # over-rate / base-rate analysis), even though it never affects P&L.
        if _settle_line_history(player_lower, args.actual, date=args.date or None):
            print(f"\n  [MARKET]  {args.player:<20}  actual={args.actual:.1f}  "
                  f"→ settled line_history (not a bet)\n")
            return
        print(f"\n  No pending pick or logged line for '{args.player}'.")
        known = sorted({r['date'] for r in rows if r['player'].lower() == player_lower})
        if known:
            print(f"  Dates on record: {', '.join(known)}")
        print()
        return

    idx, row = max(candidates, key=lambda x: x[1]['date'])
    actual  = args.actual
    outcome = _outcome(row['rec'], float(row['line']), actual)

    rows[idx]['actual']  = f'{actual:.1f}'
    rows[idx]['outcome'] = outcome
    _save(rows)

    # Also settle the market corpus (records actual for every line, bet or not)
    in_corpus = _settle_line_history(player_lower, actual, date=row['date'])

    tag = {'WIN': 'WIN ', 'LOSS': 'LOSS', 'PUSH': 'PUSH'}[outcome]
    corpus_note = '' if in_corpus else '  (no line_history row)'
    print(f"\n  [{tag}]  {row['player']:<20}  {row['rec']} {float(row['line']):.1f}  "
          f"actual={actual:.1f}  ({row['date']}){corpus_note}\n")


def cmd_stats(args) -> None:
    rows     = _load()
    settled  = [r for r in rows if r['outcome'] in ('WIN', 'LOSS', 'PUSH')]
    pending  = [r for r in rows if r['outcome'] == 'PENDING']

    print()
    print('  ════════════════════════════════════════════════')
    print('   RESULTS TRACKER — Valorant Kill Line Bets')
    print('  ════════════════════════════════════════════════')

    if not settled:
        print(f'   No settled results yet.  ({len(pending)} pending)')
        print()
        return

    wins   = sum(1 for r in settled if r['outcome'] == 'WIN')
    losses = sum(1 for r in settled if r['outcome'] == 'LOSS')
    pushes = sum(1 for r in settled if r['outcome'] == 'PUSH')
    total  = wins + losses

    win_rate = wins / total * 100 if total else 0
    pnl      = wins * 100 - losses * 110  # units at -110

    print(f'   Settled : {len(settled)}   {wins}W / {losses}L / {pushes}P')
    print(f'   Pending : {len(pending)}')
    print(f'   Win rate: {win_rate:.1f}%   (break-even at -110: 52.4%)')
    pnl_sign = '+' if pnl >= 0 else ''
    print(f'   P&L     : {pnl_sign}{pnl:.0f} units  ($1/unit, -110 odds)')

    # --- by edge band ---
    bands = [
        ('10–20%',  10,  20),
        ('20–30%',  20,  30),
        ('30–40%',  30,  40),
        ('40%+ ',   40, 999),
    ]
    band_rows = [(label, [r for r in settled if lo <= float(r['edge_pct']) < hi])
                 for label, lo, hi in bands]
    band_rows = [(l, b) for l, b in band_rows if b]

    if band_rows:
        print()
        print('   By edge band:')
        for label, band in band_rows:
            bw    = sum(1 for r in band if r['outcome'] == 'WIN')
            bl    = sum(1 for r in band if r['outcome'] == 'LOSS')
            bpush = sum(1 for r in band if r['outcome'] == 'PUSH')
            brate = bw / (bw + bl) * 100 if (bw + bl) else 0
            bpnl  = bw * 100 - bl * 110
            sign  = '+' if bpnl >= 0 else ''
            print(f'     {label}  {bw}W/{bl}L/{bpush}P  '
                  f'{brate:.0f}%  ({sign}{bpnl:.0f}u)')

    # --- by OVER / UNDER ---
    print()
    print('   OVER vs UNDER:')
    for rec in ('OVER', 'UNDER'):
        grp = [r for r in settled if r['rec'] == rec]
        if not grp:
            continue
        gw = sum(1 for r in grp if r['outcome'] == 'WIN')
        gl = sum(1 for r in grp if r['outcome'] == 'LOSS')
        grate = gw / (gw + gl) * 100 if (gw + gl) else 0
        print(f'     {rec:<6}  {gw}W/{gl}L  {grate:.0f}%')

    print()


def cmd_history(args) -> None:
    rows = _load()
    if not rows:
        print('\n  No results on record yet.\n')
        return

    n = getattr(args, 'n', 0) or 0
    display = rows[-n:] if n else rows

    print()
    print(f"  {'Date':<12} {'Player':<20} {'Line':>8} {'Pred':>8} "
          f"{'Edge':>7}  {'Rec':<6} {'Actual':>8}  {'Outcome'}")
    print('  ' + '-' * 82)
    for r in display:
        actual_str = f"{float(r['actual']):.1f}" if r.get('actual') else '—'
        print(f"  {r['date']:<12} {r['player']:<20} "
              f"{float(r['line']):>8.1f} {float(r['pred_total']):>8.1f} "
              f"{float(r['edge_pct']):>6.1f}%  {r['rec']:<6} "
              f"{actual_str:>8}  {r['outcome']}")
    print()


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def cmd_slate(args) -> None:
    """Per-slate OVER rate vs our results — 'do we beat the night or ride it?'

    For each day: the slate-wide OVER rate (what % of the WHOLE board cleared),
    our bets' win rate, and a SELECTION EDGE = our wins minus the wins we'd
    expect from betting the same DIRECTION on random players (i.e. the slate's
    base rate). Positive selection edge = our pick-selection beats blind
    directional betting; correlation of our win rate with the slate OVER rate
    tells us how much we're just riding over-heavy nights.
    """
    rows = [r for r in _load_file(LINE_HISTORY_FILE)
            if r.get('line_result') in ('OVER', 'UNDER', 'PUSH')]
    if not rows:
        print('\n  No settled market lines yet — settle slates first.\n')
        return

    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r['date']].append(r)

    print('\n  ' + '=' * 74)
    print('   SLATE OVER-RATE vs OUR RESULTS  (do we beat the night, or ride it?)')
    print('  ' + '=' * 74)
    print(f"  {'date':<12}{'lines':>6}{'OVER%':>7}{'bets':>6}{'win%':>7}"
          f"{'ourOVER%':>9}{'selEdge':>9}")
    print('  ' + '-' * 74)

    over_rates, win_rates = [], []
    T_bets = T_wins = 0
    T_exp = 0.0
    for date in sorted(by):
        rs = by[date]
        dec = [r for r in rs if r['line_result'] in ('OVER', 'UNDER')]
        if not dec:
            continue
        over_rate = sum(r['line_result'] == 'OVER' for r in dec) / len(dec)
        # "our bet" = recommended (bet_rec BET *and* edge ≥ 3pt threshold we act on)
        bets = [r for r in rs if r.get('bet_rec') == 'BET'
                and float(r.get('edge_pct') or 0) >= 3.0
                and r['line_result'] in ('OVER', 'UNDER')]
        nb = len(bets)
        wins = sum(r['rec'] == r['line_result'] for r in bets)
        our_over = (sum(r['rec'] == 'OVER' for r in bets) / nb) if nb else 0.0
        # expected wins if we'd bet the same DIRECTION on random players
        exp = sum(over_rate if r['rec'] == 'OVER' else (1 - over_rate) for r in bets)
        wr = wins / nb if nb else None
        sel = (wins - exp) if nb else None
        wr_s = f'{wr*100:>5.0f}%' if wr is not None else '   — '
        sel_s = f'{sel:+.2f}' if sel is not None else '   —'
        print(f"  {date:<12}{len(dec):>6}{over_rate*100:>6.0f}%{nb:>6}{wr_s:>7}"
              f"{our_over*100:>8.0f}%{sel_s:>9}")
        if nb:
            over_rates.append(over_rate); win_rates.append(wr)
            T_bets += nb; T_wins += wins; T_exp += exp

    print('  ' + '-' * 74)
    if T_bets:
        print(f"  TOTAL bets {T_bets}: {T_wins}W ({T_wins/T_bets*100:.0f}%)  "
              f"vs base-rate expectation {T_exp:.1f}  →  "
              f"selection edge {T_wins - T_exp:+.1f} wins ({(T_wins-T_exp)/T_bets*100:+.1f} pts)")
    r = _pearson(over_rates, win_rates)
    if r is not None:
        print(f"  corr(slate OVER-rate, our win-rate) = {r:+.2f}  "
              f"({'RIDING the night' if r > 0.5 else 'some independence'} — need more slates)")
    else:
        print(f"  (need ≥3 fully-settled slates for the ride-the-night correlation; "
              f"have {len(over_rates)})")
    print(f"\n  NOTE: only fully-settled slates are accurate — settle NON-bet lines too\n"
          f"  (results_tracker result <player> <kills> now settles market lines even if not a bet).\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Results tracker for PrizePicks Valorant kill line bets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python results_tracker.py pending
  python results_tracker.py result blowz 31
  python results_tracker.py result xavi8k 28 --date 2026-06-06
  python results_tracker.py stats
  python results_tracker.py history --n 20
""",
    )

    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('pending', help='Show picks awaiting results')

    p_res = sub.add_parser('result', help='Enter actual kills for a player')
    p_res.add_argument('player', help='Player name (case-insensitive)')
    p_res.add_argument('actual', type=float, help='Actual kills M1+M2 combined')
    p_res.add_argument('--date', default=None, help='YYYY-MM-DD (default: today)')

    sub.add_parser('stats', help='Running win rate and P&L')

    sub.add_parser('slate', help='Per-slate OVER rate vs our results (ride-the-night check)')

    p_hist = sub.add_parser('history', help='Full result log')
    p_hist.add_argument('--n', type=int, default=0, metavar='N',
                        help='Show only last N rows (default: all)')

    args = parser.parse_args()

    dispatch = {
        'pending': cmd_pending,
        'result':  cmd_result,
        'stats':   cmd_stats,
        'slate':   cmd_slate,
        'history': cmd_history,
    }

    fn = dispatch.get(args.cmd)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
