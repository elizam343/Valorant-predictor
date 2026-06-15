"""Calibration audit — do our P(win) estimates match reality? (#8 / rarga red flag)

For every SETTLED line (bet or not), compare the model's predicted P(win) for its
recommended side against whether that side actually won (rec == line_result).
Buckets by confidence → reliability curve. If predicted >> actual in the high
buckets, the distributional model is OVERCONFIDENT (the rarga problem) and we
should widen thin-sample distributions / regress extreme P(win) toward 0.5.

Run: python calibration_audit.py
"""
import csv, os
from collections import defaultdict

LH = os.path.join(os.path.dirname(__file__), 'line_history.csv')
BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]


def main():
    rows = []
    for r in csv.DictReader(open(LH)):
        if r.get('line_result') not in ('OVER', 'UNDER'):
            continue
        try:
            pw = float(r['p_win'])
        except (ValueError, TypeError, KeyError):
            continue
        rows.append((pw, r['rec'] == r['line_result'], r))
    if not rows:
        print('No settled lines.'); return

    print(f'\n  CALIBRATION — {len(rows)} settled lines across '
          f'{len(set(r[2]["date"] for r in rows))} slates\n')
    print(f'  {"conf bucket":<14}{"n":>4}{"pred":>8}{"actual":>8}{"gap":>8}')
    print('  ' + '-' * 42)
    tot_pred = tot_act = 0.0
    for lo, hi in BUCKETS:
        b = [(pw, won) for pw, won, _ in rows if lo <= pw < hi]
        if not b:
            continue
        n = len(b)
        pred = sum(pw for pw, _ in b) / n
        act = sum(w for _, w in b) / n
        tot_pred += sum(pw for pw, _ in b); tot_act += sum(w for _, w in b)
        flag = '  <-- overconfident' if pred - act > 0.10 else ''
        print(f'  {f"{lo:.2f}-{hi:.2f}":<14}{n:>4}{pred*100:>7.0f}%{act*100:>7.0f}%{(pred-act)*100:>+7.0f}{flag}')
    print('  ' + '-' * 42)
    n = len(rows)
    print(f'  {"ALL":<14}{n:>4}{tot_pred/n*100:>7.0f}%{tot_act/n*100:>7.0f}%{(tot_pred-tot_act)/n*100:>+7.0f}')

    # focused look at our highest-confidence picks (the rarga concern)
    hi = sorted(rows, key=lambda x: -x[0])[:8]
    print('\n  Top-8 most confident picks (pred / won):')
    for pw, won, r in hi:
        print(f'    {r["player"]:<12} {r["date"]}  pred {pw*100:>3.0f}%  '
              f"rec {r['rec']:<5} actual {r.get('actual','?'):>5} → {'WIN' if won else 'LOSS'}")
    over_all = tot_pred - tot_act
    print(f"\n  >> {'OVERCONFIDENT' if over_all/n > 0.05 else 'roughly calibrated'}: "
          f"predicted {tot_pred/n*100:.0f}% vs actual {tot_act/n*100:.0f}% "
          f"(n={n}, still small).")


if __name__ == '__main__':
    main()
