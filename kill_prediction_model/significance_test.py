"""Statistical significance of the betting record + the calibration finding (#8).

Two separate questions, two separate tests:
  1. Is our win rate significantly below the -110 break-even (52.4%)? (binomial)
     → with n this small, almost certainly NOT — we can't conclude "no edge" yet.
  2. Is the high-confidence OVERCONFIDENCE real or noise? Pool p_win>=0.60 picks
     and test observed wins vs the model's own expectation. → the gap may be big
     enough to be significant even at small n.
Plus a power calc: how many bets to detect a real edge?

Run: python significance_test.py
"""
import csv, os, math

BR = os.path.join(os.path.dirname(__file__), 'bet_results.csv')
LH = os.path.join(os.path.dirname(__file__), 'line_history.csv')
BREAKEVEN = 0.5238

try:
    from scipy.stats import binomtest, norm
    def binom_p(k, n, p, alt):
        return binomtest(k, n, p, alternative=alt).pvalue
    def z_ppf(q):
        return norm.ppf(q)
except ImportError:           # manual fallback
    def _binom_cdf(k, n, p):
        return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(0, k+1))
    def binom_p(k, n, p, alt):
        if alt == 'less':    return _binom_cdf(k, n, p)
        if alt == 'greater': return 1 - _binom_cdf(k-1, n, p)
        return min(1.0, 2*min(_binom_cdf(k, n, p), 1-_binom_cdf(k-1, n, p)))
    def z_ppf(q):             # rational approx (Acklam)
        a=[-39.69,220.95,-275.93,138.36,-30.66,2.506]; b=[-54.48,161.59,-155.70,66.81,-13.28]
        if q<0.5: return -z_ppf(1-q)
        t=math.sqrt(-2*math.log(1-q))
        return t-( (2.515517+0.802853*t+0.010328*t*t)/(1+1.432788*t+0.189269*t*t+0.001308*t*t*t))


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)


def main():
    # ---- 1. betting record ----
    rec = [r for r in csv.DictReader(open(BR)) if r.get('outcome') in ('WIN', 'LOSS')]
    n = len(rec); w = sum(r['outcome'] == 'WIN' for r in rec)
    wr = w/n
    lo, hi = wilson(w, n)
    print(f'\n=== 1. BETTING RECORD ({w}W/{n-w}L, n={n}) ===')
    print(f'  win rate {wr*100:.1f}%   95% CI [{lo*100:.0f}%, {hi*100:.0f}%]')
    print(f'  break-even = {BREAKEVEN*100:.1f}%   coin-flip = 50%')
    p_be = binom_p(w, n, BREAKEVEN, 'less')
    p_50 = binom_p(w, n, 0.50, 'two-sided')
    print(f'  P(win rate this low if truly break-even) = {p_be:.2f}  '
          f"→ {'SIGNIFICANT' if p_be < 0.05 else 'NOT significant — cannot conclude we lack edge'}")
    print(f'  vs 50% coin-flip: two-sided p = {p_50:.2f}  '
          f"→ {'differs' if p_50 < 0.05 else 'indistinguishable from random'}")

    # power: bets needed to detect a real edge at 80% power, one-sided 5%
    za, zb = z_ppf(0.95), z_ppf(0.80)
    print('\n  Power — bets needed to PROVE an edge (80% power, α=.05):')
    for tp in (0.55, 0.57, 0.60):
        nn = ((za+zb)**2 * tp*(1-tp)) / (tp-BREAKEVEN)**2
        print(f'    to detect a true {tp*100:.0f}% win rate ({(tp-BREAKEVEN)*100:+.1f}pt edge): ~{nn:.0f} bets')

    # ---- 2. calibration overconfidence (pool p_win >= 0.60) ----
    rows = []
    for r in csv.DictReader(open(LH)):
        if r.get('line_result') not in ('OVER', 'UNDER'):
            continue
        try:
            pw = float(r['p_win'])
        except (ValueError, TypeError, KeyError):
            continue
        rows.append((pw, r['rec'] == r['line_result']))
    hi_conf = [(pw, won) for pw, won in rows if pw >= 0.60]
    nh = len(hi_conf); wh = sum(won for _, won in hi_conf)
    exp = sum(pw for pw, _ in hi_conf)        # model's expected wins (Poisson-binomial mean)
    print(f'\n=== 2. OVERCONFIDENCE TEST (picks with p_win >= 60%, n={nh}) ===')
    print(f'  model expected ~{exp:.1f} wins ({exp/nh*100:.0f}%)   actual {wh} wins ({wh/nh*100:.0f}%)')
    p_oc = binom_p(wh, nh, exp/nh, 'less')
    print(f'  P(this few wins if model were calibrated) = {p_oc:.3f}  '
          f"→ {'SIGNIFICANT — overconfidence is REAL' if p_oc < 0.05 else 'not significant at this n'}")


if __name__ == '__main__':
    main()
