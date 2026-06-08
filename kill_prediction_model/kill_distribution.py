#!/usr/bin/env python3
"""
Distributional kill modelling + bet sizing (roadmap step 4).

The old pipeline predicted a point estimate of per-map kills, multiplied by 2
for the series, and compared a per-map line (line/2) to per-map history. That
mishandles the variance of a 2-map SUM and turns the classifier score (an
uncalibrated number) into a fake "edge".

This module instead models the distribution of the **series total** directly:

  1. Take the player's empirical per-map kill history and shift it so its mean
     equals the model's context-adjusted per-map prediction (μ̂). This keeps the
     real shape (spread, skew) while moving the centre to reflect this matchup.
  2. Convolve that distribution with itself (sum of N independent maps) to get
     the distribution of the series total — the thing the PrizePicks line is on.
  3. P(OVER) = P(series total > line). Because it comes straight from a
     distribution, it's a genuine probability, not a classifier score that needs
     post-hoc calibration.

From the win probability we derive the real betting edge (p_win − break-even)
and a fractional-Kelly stake. Players with too little history fall back to a
Normal(μ̂, σ) model using the regression's per-map kill std.

Independence across maps is assumed (corr=0); a positive same-series correlation
would widen the tails slightly — left as a future refinement.
"""

import numpy as np

# Single-leg -110 abstraction, consistent with the rest of the codebase.
# (PrizePicks is really a correlated parlay product — modelling that joint is a
# separate roadmap item; here each leg is treated as an independent -110 bet.)
DEFAULT_BREAK_EVEN = 0.524
DEC_ODDS_MINUS_110 = 1.0 + 100.0 / 110.0   # ≈ 1.9091

MIN_HIST_FOR_BOOTSTRAP = 8                  # below this, use the Normal fallback
DEFAULT_SIMS = 20000

# How far (kills/map) the regression's context adjustment may move a player from
# their OWN historical per-map mean. The player's history is the trustworthy
# anchor for their *level*; μ̂ supplies the *context nudge*. Capping this stops a
# biased μ̂ (MAE ~4.7/map) from manufacturing huge fake edges. Heuristic for now
# — should be tuned against the backtest corpus once results accumulate.
DEFAULT_SHIFT_CAP = 2.0


def series_sum_samples(mu_per_map: float, sigma_per_map: float,
                       hist_per_map=None, n_maps: int = 2,
                       n_sims: int = DEFAULT_SIMS, seed: int = 0,
                       shift_cap: float = DEFAULT_SHIFT_CAP) -> np.ndarray:
    """Monte-Carlo samples of the N-map kill SUM.

    Bootstraps the player's real per-map history (keeps its shape/spread) and
    recentres it toward the model's μ̂, but only by up to ``shift_cap`` kills so
    a hot/cold μ̂ can't override the player's own level. Falls back to
    Normal(μ̂, σ) when history is too thin to bootstrap. Seeded → deterministic.
    """
    rng = np.random.default_rng(seed)
    hist = (np.asarray(hist_per_map, dtype=float)
            if hist_per_map is not None and len(hist_per_map) else None)

    if hist is not None and len(hist) >= MIN_HIST_FOR_BOOTSTRAP:
        shift = mu_per_map - hist.mean()           # nudge toward μ̂ …
        if shift_cap is not None:                  # … but only within the cap
            shift = float(np.clip(shift, -shift_cap, shift_cap))
        draws = rng.choice(hist, size=(n_sims, n_maps), replace=True) + shift
    else:
        sigma = max(float(sigma_per_map), 1e-6)
        draws = rng.normal(mu_per_map, sigma, size=(n_sims, n_maps))

    return draws.sum(axis=1)


def prob_over(line: float, mu_per_map: float, sigma_per_map: float,
              hist_per_map=None, n_maps: int = 2, **kw) -> float:
    """P(series total > line)."""
    s = series_sum_samples(mu_per_map, sigma_per_map, hist_per_map, n_maps, **kw)
    return float((s > line).mean())


def kelly_stake(p_win: float, dec_odds: float = DEC_ODDS_MINUS_110,
                fraction: float = 0.25, cap: float = 0.05) -> float:
    """Fractional-Kelly stake as a fraction of bankroll.

    Full Kelly f* = (b·p − q) / b, b = dec_odds − 1. We bet `fraction` of that
    (default quarter-Kelly), clamp negatives to 0, and cap exposure per leg.
    """
    b = dec_odds - 1.0
    q = 1.0 - p_win
    f_star = (b * p_win - q) / b
    return float(np.clip(fraction * f_star, 0.0, cap))


def evaluate_pick(line: float, mu_per_map: float, sigma_per_map: float,
                  hist_per_map=None, n_maps: int = 2, clf_p_over=None,
                  break_even: float = DEFAULT_BREAK_EVEN,
                  kelly_fraction: float = 0.25, kelly_cap: float = 0.05,
                  conflict_margin: float = 0.10,
                  shift_cap: float = DEFAULT_SHIFT_CAP) -> dict:
    """Score a single OVER/UNDER pick distributionally.

    Returns p_over, the recommended side, win prob, edge (percentage POINTS over
    break-even), Kelly stake, a BET / NO BET flag, and a classifier cross-check.
    A *confident* classifier disagreement vetoes the bet; a mild one is a note.
    """
    p_over = prob_over(line, mu_per_map, sigma_per_map, hist_per_map, n_maps,
                       shift_cap=shift_cap)
    rec    = 'OVER' if p_over >= 0.5 else 'UNDER'
    p_win  = p_over if rec == 'OVER' else 1.0 - p_over
    edge_pts = (p_win - break_even) * 100.0
    stake    = kelly_stake(p_win, fraction=kelly_fraction, cap=kelly_cap)

    bet_rec, reason, cross_note = 'BET', '', ''
    if p_win <= break_even:
        bet_rec = 'NO BET'
        reason  = f'no edge (p_win {p_win:.0%} ≤ {break_even:.0%})'

    if clf_p_over is not None:
        clf_dir = 'OVER' if clf_p_over >= 0.5 else 'UNDER'
        if clf_dir != rec:
            cross_note = f'clf {clf_dir} {clf_p_over:.0%}'
            if abs(clf_p_over - 0.5) > conflict_margin and bet_rec == 'BET':
                bet_rec = 'NO BET'
                reason  = f'conflict (clf {clf_dir} {clf_p_over:.0%})'

    return {
        'p_over': p_over, 'rec': rec, 'p_win': p_win,
        'edge_pts': edge_pts, 'stake': stake,
        'bet_rec': bet_rec, 'filter_reason': reason,
        'clf_p_over': clf_p_over, 'cross_note': cross_note,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(1)
    # Synthetic player: per-map kills ~ N(17, 5), 200 maps of history
    hist = rng.normal(17, 5, size=200)

    # 1) Uncapped convolution recentres fully on μ̂ and sums two maps
    s = series_sum_samples(20.0, 5.0, hist, n_maps=2, shift_cap=None)
    assert abs(s.mean() - 40.0) < 0.4, s.mean()          # 2 × μ̂ (hist mean ~17 + shift 3)
    print(f'  uncapped sum mean ≈ {s.mean():.2f}  (expect ~40)')
    print(f'  sum std           ≈ {s.std():.2f}  (≈ √2 × per-map std)')

    # 1b) shift_cap limits how far μ̂ moves the centre off the player's own mean
    sc = series_sum_samples(20.0, 5.0, hist, n_maps=2, shift_cap=2.0)
    assert abs(sc.mean() - 2 * (hist.mean() + 2.0)) < 0.4, sc.mean()   # hist_mean+2 per map
    print(f'  capped(2) sum mean ≈ {sc.mean():.2f}  (expect ~{2*(hist.mean()+2):.1f}, < 40)')

    # 2) P(OVER) is monotonically decreasing in the line
    ps = [prob_over(L, 20.0, 5.0, hist) for L in (30, 35, 40, 45, 50)]
    assert all(a >= b for a, b in zip(ps, ps[1:])), ps
    print('  P(over) by line 30→50:', [f'{p:.2f}' for p in ps])

    # 3) Kelly: positive only with edge, capped
    assert kelly_stake(0.50) == 0.0
    assert 0 < kelly_stake(0.60) <= 0.05
    assert kelly_stake(0.99) == 0.05                      # cap
    print(f'  kelly p=.55 → {kelly_stake(0.55):.3f}   p=.65 → {kelly_stake(0.65):.3f}')

    # 4) evaluate_pick: a strong line we clear → BET OVER with stake
    ev = evaluate_pick(line=30.0, mu_per_map=20.0, sigma_per_map=5.0, hist_per_map=hist)
    assert ev['rec'] == 'OVER' and ev['bet_rec'] == 'BET' and ev['stake'] > 0, ev
    print('  evaluate_pick(line=30, μ̂=20):', {k: (round(v, 3) if isinstance(v, float) else v)
                                               for k, v in ev.items() if k != 'clf_p_over'})

    # 5) Confident classifier disagreement vetoes
    ev2 = evaluate_pick(line=30.0, mu_per_map=20.0, sigma_per_map=5.0,
                        hist_per_map=hist, clf_p_over=0.20)
    assert ev2['bet_rec'] == 'NO BET' and 'conflict' in ev2['filter_reason'], ev2
    print('  conflict veto:', ev2['filter_reason'])

    print('\n  ✓ all self-tests passed')
