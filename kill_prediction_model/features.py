"""Single source of truth for the model's feature columns.

Historically this list was duplicated in 5 places (bet_slate, model_comparison,
enhanced_data_loader's local list, db_data_loader, backtester). They had to stay
byte-identical or the model silently predicted on mis-ordered features
(train/serve skew) — and adding one feature meant editing each by hand (#6).

Training (model_comparison + enhanced_data_loader) and serving (bet_slate) now
all import FEATURE_COLS from here, so they cannot drift. Add a feature in ONE
place: append it here, ensure both the loader produces it and the cache serves it.

CLF_EXTRA = the two extra columns the OVER/UNDER classifier gets on top of the
regression features (the line itself + the player's historical hit rate at it).
"""

FEATURE_COLS = [
    # Career averages from vlr_players.db
    'db_rating', 'db_average_combat_score', 'db_kill_deaths',
    'db_kills_per_round', 'db_assists_per_round',
    'db_first_kills_per_round', 'db_first_deaths_per_round',
    # Match context
    'team_strength',
    'opponent_team_strength',
    'opponent_kills_allowed_per_map',
    # Rolling recent form
    'recent_avg_kills', 'recent_avg_rating',
    'recent_avg_kills_3',         # last 3 maps (faster recency signal)
    'form_slope',                 # kills trend (last 5 maps)
    'rating_form_slope',          # efficiency trajectory (rating trend)
    'days_since_last_match',      # rest days — freshness / preparation
    # Head-to-head history
    'h2h_avg_kills',
    'h2h_data_exists',
    # Map familiarity
    'player_map_avg_kills',
    # Expected rounds from team-matchup history
    'avg_rounds_vs_opponent',
    # Kill variance
    'kill_std',
    # Agent role features
    'agent_role_ordinal',
    'is_duelist',
    'player_agent_avg_kills',
]

# Classification gets two extra features: the kill line + player historical hit rate
CLF_EXTRA = ['synthetic_line', 'player_hit_rate_at_line']


if __name__ == '__main__':
    # Parity self-test: train-side and serve-side must import THIS exact list.
    import importlib
    ok = True
    for mod_name in ('model_comparison', 'bet_slate'):
        m = importlib.import_module(mod_name)
        if list(getattr(m, 'FEATURE_COLS')) != FEATURE_COLS:
            print(f'  ✗ {mod_name}.FEATURE_COLS DIFFERS from features.FEATURE_COLS')
            ok = False
        else:
            print(f'  ✓ {mod_name}.FEATURE_COLS matches ({len(FEATURE_COLS)} features)')
    print('\n  ✓ feature lists are unified' if ok else '\n  ✗ train/serve skew detected')
    raise SystemExit(0 if ok else 1)
