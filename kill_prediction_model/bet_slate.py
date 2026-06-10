#!/usr/bin/env python3
"""
Daily bet slate generator for PrizePicks Valorant kill lines.

Fetches all active kill projections from PrizePicks, runs the trained
regression + classifier models (whichever won model_comparison) on each
player, and prints a ranked table of OVER/UNDER recommendations sorted by edge.

Usage
-----
  # Basic: fetch live lines, use historical averages for context
  python bet_slate.py

  # Only show bets where model disagrees by ≥ 15%
  python bet_slate.py --min-edge 15

  # Provide matchup context (team, opponent, maps, agent) for better accuracy
  python bet_slate.py --context context.json

  # Limit match history loaded for cache (faster startup, less accurate)
  python bet_slate.py --cache-matches 1000

Context file format (context.json)
-----------------------------------
  {
    "aspas":    {"team": "MIBR",   "opponent": "FUR",   "maps": ["Ascent", "Bind"], "agent": "Jett"},
    "something": {"team": "Team A", "opponent": "Team B", "maps": ["Haven"]}
  }
  All fields are optional — anything omitted uses historical averages.
"""

import sys
import os
import json
import argparse
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import joblib
warnings.filterwarnings('ignore', message='X does not have valid feature names')

sys.path.insert(0, os.path.dirname(__file__))
from enhanced_data_loader import EnhancedDataLoader, AGENT_ROLES, UNKNOWN_ROLE
from kill_line_fetcher import PrizePicksClient
from results_tracker import save_slate as _save_slate
from results_tracker import log_line_history as _log_line_history
import kill_distribution as kd

_ALIASES_FILE = os.path.join(os.path.dirname(__file__), 'name_aliases.json')


def _load_aliases() -> Dict[str, str]:
    """Load PP→VLR name mappings from name_aliases.json (lowercase keys)."""
    if not os.path.exists(_ALIASES_FILE):
        return {}
    with open(_ALIASES_FILE) as f:
        data = json.load(f)
    return {k.lower(): v.lower() for k, v in data.items() if not k.startswith('_')}

BREAKEVEN   = 0.5238   # -110 odds
# Min historical hit rate at the PP line to recommend OVER
BASE_RATE_THRESHOLD = 0.45

# Role-conditioned variance (v1): bootstrap the kill distribution from the
# player's ON-AGENT history when they're locked on a well-sampled agent, so a
# flex player's blended career shape doesn't misprice P(over). See scope notes.
MIN_AGENT_MAPS  = 20    # need at least this many maps on the agent to trust its shape
MIN_AGENT_SHARE = 0.40  # if the agent was GUESSED, only filter when it's clearly their main
# Feature list lives in features.py — single source of truth (see #6).
from features import FEATURE_COLS
DEFAULTS = {
    'db_rating':                      1.0,
    'db_average_combat_score':      193.0,
    'db_kill_deaths':                 0.92,
    'db_kills_per_round':             0.67,
    'db_assists_per_round':           0.27,
    'db_first_kills_per_round':       0.09,
    'db_first_deaths_per_round':      0.10,
    'team_strength':                  1.0,
    'opponent_team_strength':         1.0,
    'opponent_kills_allowed_per_map': 13.0,
    'recent_avg_kills':              13.0,
    'recent_avg_rating':              1.0,
    'recent_avg_kills_3':            13.0,
    'form_slope':                     0.0,
    'rating_form_slope':              0.0,
    'days_since_last_match':          7.0,
    'h2h_avg_kills':                 13.0,
    'h2h_data_exists':                0.0,
    'player_map_avg_kills':          13.0,
    'avg_rounds_vs_opponent':        20.0,
    'kill_std':                       5.0,
    'agent_role_ordinal':        UNKNOWN_ROLE,
    'is_duelist':                     0.0,
    'player_agent_avg_kills':        13.0,
    'player_hit_rate_at_line':        0.5,
}


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

class PlayerCache:
    """Lightweight lookup tables built from scraped match history."""

    def __init__(self, limit: int = 2000):
        loader = EnhancedDataLoader()

        print(f'Loading player cache from {limit} matches...')
        matches = loader.load_scraped_matches(limit=limit)

        team_ratings:        dict = defaultdict(list)
        player_ratings:      dict = defaultdict(list)
        player_kills:        dict = defaultdict(list)   # chronological, for form_slope
        player_map_kills:    dict = defaultdict(list)
        player_agent_kills:  dict = defaultdict(list)
        player_agent_freq:   dict = defaultdict(int)    # (player, agent) → appearances
        team_map_freq:       dict = defaultdict(int)    # (team, map) → appearances
        h2h_kills:           dict = defaultdict(list)   # (player, opponent_team) → kills
        opp_allowed:         dict = defaultdict(list)   # opponent_team → kills scored against them

        for match in matches:
            # Build a quick set of opponent teams per player team in this match
            teams_in_match = list({p.team for p in match.players if p.team})
            for p in match.players:
                pname = p.name.lower()
                opp_team = next((t for t in teams_in_match if t != p.team), '')
                if p.rating > 0:
                    team_ratings[p.team].append(p.rating)
                    player_ratings[pname].append(p.rating)
                if p.kills > 0:
                    player_kills[pname].append(p.kills)
                    if p.map_name:
                        player_map_kills[(pname, p.map_name)].append(p.kills)
                        if p.team:
                            team_map_freq[(p.team, p.map_name)] += 1
                    if p.agent:
                        agent_key = p.agent.lower()
                        player_agent_kills[(pname, agent_key)].append(p.kills)
                        player_agent_freq[(pname, agent_key)] += 1
                    if opp_team:
                        h2h_kills[(pname, opp_team)].append(p.kills)
                        opp_allowed[opp_team].append(p.kills)

        self.team_strength      = {t: float(np.mean(v))       for t, v in team_ratings.items()}
        self.player_rating      = {p: float(np.mean(v[-10:])) for p, v in player_ratings.items()}
        self.player_rating_hist = dict(player_ratings)         # raw list for rating_form_slope
        self.player_kills       = {p: float(np.mean(v[-10:])) for p, v in player_kills.items()}
        self.player_kill_hist   = dict(player_kills)           # raw list for form_slope + hit rate
        self.player_kill_std    = {
            p: float(np.std(v)) if len(v) > 1 else 3.0
            for p, v in player_kills.items()
        }
        self.map_kills          = {k: float(np.mean(v))        for k, v in player_map_kills.items()}
        self.agent_kills        = {k: float(np.mean(v))        for k, v in player_agent_kills.items()}
        # Raw per-(player, agent) kill lists — the empirical SHAPE used to
        # bootstrap a role-correct distribution when the player is locked on a
        # well-sampled agent (flex players' blended career hist is wrong-shaped).
        self.player_agent_hist  = {k: list(v)                  for k, v in player_agent_kills.items()}
        self.player_agent_freq  = dict(player_agent_freq)
        self.team_map_freq      = dict(team_map_freq)
        self.h2h_kills          = {k: float(np.mean(v))        for k, v in h2h_kills.items()}
        self.h2h_count_dict     = {k: len(v)                   for k, v in h2h_kills.items()}
        self.opp_allowed        = {t: float(np.mean(v))        for t, v in opp_allowed.items()}
        self.player_appearances = {p: len(v) for p, v in player_kills.items()}

        # Case-insensitive team name index so PrizePicks names match VLR names
        self._team_lower = {t.lower(): t for t in self.team_strength if t}

        # Player career stats from the DB (db_rating, db_kills_per_round, etc.)
        try:
            db_df = loader.load_player_database_stats()
            self.db_stats = {
                row['name'].lower(): row
                for _, row in db_df.iterrows()
            }
        except Exception:
            self.db_stats = {}

        # Full DB lookups — kill distributions, last match date, avg rounds per matchup
        self.db_kill_hist: Dict[str, list] = {}
        self.player_last_match: Dict[str, str] = {}
        self.team_opp_avg_rounds: Dict[tuple, float] = {}
        self.map_avg_rounds: Dict[str, float] = {}
        try:
            import sqlite3 as _sqlite3
            from datetime import date as _date
            _db   = os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db')
            _conn = _sqlite3.connect(_db)
            _cur  = _conn.cursor()

            # Kill distributions (full career — more accurate than JSON sample)
            _cur.execute("""
                SELECT LOWER(p.name), pms.kills
                FROM player_match_stats pms
                JOIN players p ON p.id = pms.player_id
                WHERE pms.kills BETWEEN 1 AND 40 AND pms.acs > 0
            """)
            _raw: Dict[str, list] = defaultdict(list)
            for _name, _k in _cur.fetchall():
                _raw[_name].append(_k)
            self.db_kill_hist = dict(_raw)

            # Last match date per player (for days_since_last_match)
            _cur.execute("""
                SELECT LOWER(p.name), MAX(m.match_date)
                FROM player_match_stats pms
                JOIN players p ON p.id = pms.player_id
                JOIN matches m ON m.id = pms.match_id
                WHERE pms.kills BETWEEN 1 AND 40 AND pms.acs > 0
                GROUP BY LOWER(p.name)
            """)
            self.player_last_match = {r[0]: r[1] for r in _cur.fetchall() if r[1]}

            # Avg estimated rounds per team-opponent pairing
            _cur.execute("""
                WITH map_totals AS (
                    SELECT t.name AS team, ot.name AS opp,
                           CAST(SUM(pms.kills) AS REAL) / 6.0 AS est_rounds
                    FROM player_match_stats pms
                    JOIN teams   t  ON t.id  = pms.team_id
                    JOIN maps    mn ON mn.id = pms.map_id
                    JOIN matches m  ON m.id  = pms.match_id
                    JOIN teams   ot ON ot.id = CASE
                        WHEN m.team1_id = pms.team_id THEN m.team2_id
                        ELSE m.team1_id END
                    WHERE pms.kills BETWEEN 1 AND 40 AND pms.acs > 0
                    GROUP BY pms.match_id, mn.map_name, t.name, ot.name
                )
                SELECT team, opp, AVG(est_rounds) FROM map_totals GROUP BY team, opp
            """)
            self.team_opp_avg_rounds = {
                (r[0].lower(), r[1].lower()): float(r[2]) for r in _cur.fetchall()
            }

            # Avg estimated rounds per map name (global fallback)
            _cur.execute("""
                WITH map_totals AS (
                    SELECT mn.map_name,
                           CAST(SUM(pms.kills) AS REAL) / 6.0 AS est_rounds
                    FROM player_match_stats pms
                    JOIN maps mn ON mn.id = pms.map_id
                    WHERE pms.kills BETWEEN 1 AND 40 AND pms.acs > 0
                    GROUP BY pms.match_id, pms.map_id, mn.map_name
                )
                SELECT map_name, AVG(est_rounds) FROM map_totals GROUP BY map_name
            """)
            self.map_avg_rounds = {r[0].lower(): float(r[1]) for r in _cur.fetchall()}

            _conn.close()
        except Exception as _e:
            print(f'  Warning: DB enrichment failed ({_e}) — using JSON cache only')

        print(f'Cache ready: {len(self.team_strength)} teams, '
              f'{len(self.player_kills)} players, '
              f'{len(self.db_stats)} career stat entries, '
              f'{len(self.db_kill_hist)} DB kill distributions')

    # -- Persistence: build the cache once, reuse it instantly thereafter -----
    def save(self, path: str) -> None:
        """Pickle the computed lookup tables so future runs skip the JSON parse."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.__dict__, path)

    @classmethod
    def from_file(cls, path: str) -> 'PlayerCache':
        """Reconstruct a cache from a saved pickle without re-running __init__."""
        obj = cls.__new__(cls)          # bypass __init__ (no JSON parsing)
        obj.__dict__ = joblib.load(path)
        return obj

    def likely_agent(self, player_name: str) -> str:
        """Return the agent this player has played most often historically."""
        candidates = {
            agent: count
            for (p, agent), count in self.player_agent_freq.items()
            if p == player_name
        }
        if not candidates:
            return ''
        return max(candidates, key=candidates.get)

    def select_kill_hist(self, player_name: str, agent: str,
                         agent_guessed: bool) -> Tuple[list, str]:
        """Pick the empirical per-map kill history to bootstrap the distribution.

        Role-conditioned variance (v1): when the player is locked on a
        well-sampled agent we use their ON-AGENT history (correct shape for a
        flexing player); otherwise we fall back to the full career distribution.
        Returns ``(hist, source)`` where source ∈ {'agent', 'career'} for logging.
        """
        career = (self.db_kill_hist.get(player_name)
                  or self.player_kill_hist.get(player_name, []))

        agent_key = (agent or '').lower()
        on_agent  = self.player_agent_hist.get((player_name, agent_key), [])
        if len(on_agent) >= MIN_AGENT_MAPS:
            # A guessed agent is just likely_agent's pick (their main); trust it
            # only when it's clearly their main to avoid filtering on a bad guess.
            if agent_guessed:
                # Share is vs AGENT-KNOWN maps, not all appearances — the JSON
                # has many agentless maps that would otherwise dilute the share.
                if not hasattr(self, '_agent_known_total'):
                    _tot: dict = defaultdict(int)
                    for (_p, _a), _n in self.player_agent_freq.items():
                        _tot[_p] += _n
                    self._agent_known_total = dict(_tot)
                total = self._agent_known_total.get(player_name, 0)
                share = len(on_agent) / total if total else 0.0
                if share >= MIN_AGENT_SHARE:
                    return on_agent, 'agent'
            else:
                return on_agent, 'agent'   # agent explicitly provided → trust it
        return career, 'career'

    def _resolve_team(self, name: str) -> str:
        """Return the canonical team name from our cache, case-insensitively."""
        return self._team_lower.get(name.lower(), name)

    def likely_maps(self, team: str, opponent: str, n: int = 2) -> List[str]:
        """
        Return the n most likely maps for this matchup based on each team's
        historical map frequency.  Scores maps by the sum of normalised
        play-rates for both teams so maps both sides favour rise to the top.
        """
        team     = self._resolve_team(team)
        opponent = self._resolve_team(opponent)
        team_maps = {m: c for (t, m), c in self.team_map_freq.items() if t == team}
        opp_maps  = {m: c for (t, m), c in self.team_map_freq.items() if t == opponent}

        all_maps = set(team_maps) | set(opp_maps)
        if not all_maps:
            return []

        team_total = sum(team_maps.values()) or 1
        opp_total  = sum(opp_maps.values()) or 1

        scores = {
            m: (team_maps.get(m, 0) / team_total) + (opp_maps.get(m, 0) / opp_total)
            for m in all_maps
        }
        return sorted(scores, key=scores.get, reverse=True)[:n]

    def features(
        self,
        player_name: str,
        team: str = '',
        opponent: str = '',
        maps: Optional[List[str]] = None,
        agent: str = '',
    ) -> Tuple[Dict, List[str]]:
        """
        Build feature dict for this player + context.
        Returns (features, missing_fields) where missing_fields lists anything
        that fell back to the global default.
        """
        f: Dict = {}
        missing: List[str] = []

        # --- Career stats from DB ---
        db = self.db_stats.get(player_name)  # pandas Series or None
        for col in ['db_rating', 'db_average_combat_score', 'db_kill_deaths',
                    'db_kills_per_round', 'db_assists_per_round',
                    'db_first_kills_per_round', 'db_first_deaths_per_round']:
            key = col.replace('db_', '')
            val = db.get(key) if db is not None else None
            if val is not None and float(val) > 0:
                f[col] = float(val)
            else:
                f[col] = DEFAULTS[col]
                missing.append(col)

        # --- Team strengths ---
        canonical_team     = self._resolve_team(team)     if team     else ''
        canonical_opponent = self._resolve_team(opponent) if opponent else ''

        ts = self.team_strength.get(canonical_team) if canonical_team else None
        f['team_strength'] = ts if ts else DEFAULTS['team_strength']
        if not ts:
            missing.append('team_strength')

        os_ = self.team_strength.get(canonical_opponent) if canonical_opponent else None
        f['opponent_team_strength'] = os_ if os_ else DEFAULTS['opponent_team_strength']
        if not os_:
            missing.append('opponent_team_strength')

        # --- Opponent defensive rating ---
        oka = self.opp_allowed.get(canonical_opponent) if canonical_opponent else None
        f['opponent_kills_allowed_per_map'] = oka if oka else DEFAULTS['opponent_kills_allowed_per_map']
        if not oka:
            missing.append('opponent_kills_allowed_per_map')

        # --- Recent form ---
        rk = self.player_kills.get(player_name)
        f['recent_avg_kills'] = rk if rk else DEFAULTS['recent_avg_kills']
        if not rk:
            missing.append('recent_avg_kills')

        rr = self.player_rating.get(player_name)
        f['recent_avg_rating'] = rr if rr else DEFAULTS['recent_avg_rating']

        # --- Last 3 maps avg (faster recency signal) ---
        hist = self.player_kill_hist.get(player_name, [])
        f['recent_avg_kills_3'] = float(np.mean(hist[-3:])) if len(hist) >= 1 else DEFAULTS['recent_avg_kills_3']

        # --- Form slope (trend of last 5 maps) ---
        if len(hist) >= 3:
            recent5 = hist[-5:]
            slope = float(np.polyfit(range(len(recent5)), recent5, 1)[0])
        else:
            slope = DEFAULTS['form_slope']
        f['form_slope'] = slope

        # --- Rating form slope (efficiency trajectory, last 5 maps) ---
        # Mirrors form_slope but on VLR rating — role/round-robust, so it tracks
        # genuine improvement/decline independent of kill volume. Same as training.
        rhist = self.player_rating_hist.get(player_name, [])
        if len(rhist) >= 3:
            recent5r = rhist[-5:]
            f['rating_form_slope'] = float(np.polyfit(range(len(recent5r)), recent5r, 1)[0])
        else:
            f['rating_form_slope'] = DEFAULTS['rating_form_slope']

        # --- Days since last match ---
        from datetime import date as _today_date
        import pandas as _pd
        last_str = self.player_last_match.get(player_name)
        if last_str:
            try:
                last_dt = _pd.to_datetime(last_str).date()
                days = (_today_date.today() - last_dt).days
                f['days_since_last_match'] = float(min(max(days, 0), 30))
            except Exception:
                f['days_since_last_match'] = DEFAULTS['days_since_last_match']
        else:
            f['days_since_last_match'] = DEFAULTS['days_since_last_match']

        # --- Head-to-head vs this opponent ---
        h2h = self.h2h_kills.get((player_name, canonical_opponent)) if canonical_opponent else None
        f['h2h_avg_kills'] = h2h if h2h else (
            self.player_kills.get(player_name) or DEFAULTS['h2h_avg_kills']
        )
        if not h2h:
            missing.append('h2h_avg_kills')

        h2h_count = self.h2h_count_dict.get((player_name, canonical_opponent), 0) if canonical_opponent else 0
        f['h2h_data_exists'] = 1.0 if h2h_count > 1 else 0.0

        # --- Kill standard deviation ---
        kstd = self.player_kill_std.get(player_name)
        f['kill_std'] = kstd if kstd is not None else DEFAULTS['kill_std']

        # --- Avg rounds vs this opponent ---
        rounds_key = (canonical_team.lower(), canonical_opponent.lower()) \
                     if canonical_team and canonical_opponent else None
        if rounds_key and rounds_key in self.team_opp_avg_rounds:
            f['avg_rounds_vs_opponent'] = self.team_opp_avg_rounds[rounds_key]
        elif maps:
            # Fall back to map-name averages if no team matchup data
            map_rounds = [self.map_avg_rounds.get(m.lower()) for m in maps]
            map_rounds = [r for r in map_rounds if r is not None]
            f['avg_rounds_vs_opponent'] = float(np.mean(map_rounds)) if map_rounds else DEFAULTS['avg_rounds_vs_opponent']
        else:
            f['avg_rounds_vs_opponent'] = DEFAULTS['avg_rounds_vs_opponent']

        # --- Map avg kills ---
        if maps:
            vals = [self.map_kills.get((player_name, m)) for m in maps]
            vals = [v for v in vals if v is not None]
            f['player_map_avg_kills'] = float(np.mean(vals)) if vals else DEFAULTS['player_map_avg_kills']
            if not vals:
                missing.append('player_map_avg_kills')
        else:
            f['player_map_avg_kills'] = (
                self.player_kills.get(player_name) or DEFAULTS['player_map_avg_kills']
            )
            missing.append('player_map_avg_kills(no maps)')

        # --- Agent features ---
        if agent:
            role = float(AGENT_ROLES.get(agent.lower().strip(), UNKNOWN_ROLE))
            f['agent_role_ordinal'] = role
            f['is_duelist'] = 1.0 if role == 3.0 else 0.0
            ak = self.agent_kills.get((player_name, agent.lower()))
            f['player_agent_avg_kills'] = ak if ak else (
                self.player_kills.get(player_name) or DEFAULTS['player_agent_avg_kills']
            )
        else:
            f['agent_role_ordinal']    = DEFAULTS['agent_role_ordinal']
            f['is_duelist']            = DEFAULTS['is_duelist']
            f['player_agent_avg_kills'] = (
                self.player_kills.get(player_name) or DEFAULTS['player_agent_avg_kills']
            )
            missing.append('agent')

        return f, missing


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def load_gbr(model_dir: str = 'models') -> Tuple:
    # Prefer the newly trained best regression model; fall back to GPU model
    base = os.path.dirname(__file__)
    for fname, key in [('best_regression_model.pkl', 'feature_cols'),
                       ('gradient_boosting_gpu_model.pkl', 'feature_columns')]:
        path = os.path.join(base, model_dir, fname)
        if os.path.exists(path):
            data = joblib.load(path)
            meta = {'name': data.get('model_name', '?'),
                    'perf': data.get('performance', {})}
            return data['model'], data['scaler'], data[key], meta
    raise FileNotFoundError('No regression model found in models/')


def load_classifier(model_dir: str = 'models') -> Optional[Tuple]:
    """Load the saved OVER/UNDER classifier. Returns None if not found."""
    path = os.path.join(os.path.dirname(__file__), model_dir, 'best_classifier_model.pkl')
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    meta = {'name': data.get('model_name', '?'),
            'perf': data.get('performance', {})}
    return data['model'], data['scaler'], data['feature_cols'], meta


def predict(model, scaler, feature_cols, feat_dict: Dict) -> float:
    """Regression: predict per-map kill count."""
    vec = [feat_dict.get(c, DEFAULTS.get(c, 0.0)) for c in feature_cols]
    scaled = scaler.transform([vec])
    return float(model.predict(scaled)[0])


def predict_proba_over(clf_model, clf_scaler, clf_cols, feat_dict: Dict,
                       per_map_line: float) -> float:
    """
    Classification: return P(OVER) given player features + the actual line.
    The classifier was trained with 'synthetic_line' as the 16th feature;
    in production we substitute the real PrizePicks per-map line.
    """
    feat = dict(feat_dict)
    feat['synthetic_line'] = per_map_line
    vec = [feat.get(c, DEFAULTS.get(c, 0.0)) for c in clf_cols]
    scaled = clf_scaler.transform([vec])
    proba = clf_model.predict_proba(scaled)[0]
    return float(proba[1])  # index 1 = OVER class


# ---------------------------------------------------------------------------
# Slate builder
# ---------------------------------------------------------------------------

def build_slate(
    cache: PlayerCache,
    model,
    scaler,
    feature_cols,
    lines: Dict[str, float],
    context: Dict[str, Dict],
    min_edge: float,
    min_appearances: int = 15,
    clf=None,
    clf_scaler=None,
    clf_cols: Optional[List[str]] = None,
    market_weight: float = kd.DEFAULT_MARKET_WEIGHT,
) -> List[Dict]:
    # PrizePicks now uses "MAPS 1-2 Kills" (2-map totals).
    # Model predicts per-map kills, so multiply by 2 to compare.
    MAPS = 2
    using_clf = clf is not None and clf_scaler is not None and clf_cols is not None
    aliases   = _load_aliases()

    skipped_debuts = []
    rows = []
    for player_raw, line in lines.items():
        # Apply name alias: PP name → VLR canonical name for cache lookups
        player = aliases.get(player_raw.lower(), player_raw)
        appearances = cache.player_appearances.get(player, 0)
        if appearances < min_appearances:
            skipped_debuts.append((player, line, appearances))
            continue
        ctx      = dict(context.get(player, {}))
        team     = ctx.get('team', '')
        opponent = ctx.get('opponent', '')

        # Auto-predict maps if not provided
        maps          = ctx.get('maps')
        maps_guessed  = False
        if not maps and team and opponent:
            maps = cache.likely_maps(team, opponent, n=2)
            if maps:
                ctx['maps']  = maps
                maps_guessed = True

        # Auto-predict agent if not provided
        agent          = ctx.get('agent', '')
        agent_guessed  = False
        if not agent:
            agent = cache.likely_agent(player)
            if agent:
                ctx['agent']   = agent
                agent_guessed  = True

        feat, missing = cache.features(
            player_name = player,
            team        = team,
            opponent    = opponent,
            maps        = maps,
            agent       = agent,
        )

        # Regression mean = context-adjusted per-map kill expectation (μ̂)
        pred_per_map = predict(model, scaler, feature_cols, feat)
        pred_total   = pred_per_map * MAPS

        # Per-map kill history — the empirical shape for the series distribution.
        # Role-conditioned (v1): prefer the player's ON-AGENT history when they're
        # locked on a well-sampled agent, else the full career distribution.
        per_map_line = line / MAPS
        hist, hist_source = cache.select_kill_hist(player, agent, agent_guessed)
        hit_rate = float((np.array(hist) > per_map_line).mean()) if hist else 0.5
        n_maps   = len(hist)

        # Classifier P(OVER) — now a secondary cross-check, not the driver.
        clf_p_over = None
        if using_clf:
            feat['player_hit_rate_at_line'] = hit_rate
            clf_p_over = predict_proba_over(clf, clf_scaler, clf_cols, feat, per_map_line)

        # Distributional scoring: model the 2-map SUM directly, derive a real
        # win probability, edge (points over break-even), and Kelly stake.
        sigma_pm = feat.get('kill_std') or DEFAULTS['kill_std']
        ev = kd.evaluate_pick(
            line, mu_per_map=pred_per_map, sigma_per_map=sigma_pm,
            hist_per_map=hist, n_maps=MAPS, clf_p_over=clf_p_over,
            break_even=BREAKEVEN, market_weight=market_weight,
        )

        # Market-shrunk μ̂ is what actually drives the bet — report that, not the
        # raw (hot) regression output, so display/logging match the decision.
        adj_per_map = ev['mu_adj']
        adj_total   = adj_per_map * MAPS

        rows.append({
            'player':        player_raw,
            'player_lookup': player,
            'line':          line,
            'pred_per_map':  adj_per_map,
            'pred_total':    adj_total,
            'pred_per_map_raw': pred_per_map,
            'prob_over':     ev['p_over'],      # distributional P(OVER), primary
            'p_win':         ev['p_win'],
            'edge':          ev['edge_pts'],    # percentage POINTS over break-even
            'stake':         ev['stake'],       # fraction of bankroll (¼-Kelly)
            'rec':           ev['rec'],
            'bet_rec':       ev['bet_rec'],
            'filter_reason': ev['filter_reason'],
            'clf_p_over':    clf_p_over,
            'cross_note':    ev['cross_note'],
            'hit_rate':      hit_rate,
            'n_maps':        n_maps,
            'hist_source':   hist_source,   # 'agent' | 'career' — for calibration A/B
            'conflict':      bool(ev['cross_note']),
            'using_clf':     using_clf,
            'missing':       missing,
            'context':       ctx,
            'maps_guessed':  maps_guessed,
            'agent_guessed': agent_guessed,
        })

    rows.sort(key=lambda r: r['edge'], reverse=True)
    return rows, skipped_debuts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _ctx_label(row: Dict) -> str:
    ctx   = row['context']
    parts = []
    if ctx.get('opponent'):
        parts.append(f"vs {ctx['opponent']}")
    if ctx.get('maps'):
        prefix = '~' if row.get('maps_guessed') else ''
        parts.append(f"{prefix}maps:{'/'.join(ctx['maps'])}")
    if ctx.get('agent'):
        prefix = '~' if row.get('agent_guessed') else ''
        parts.append(f"{prefix}agent:{ctx['agent']}")
    # Only flag truly missing fields (not ones we filled in via guessing)
    truly_missing = [m for m in row['missing'] if 'map' not in m and 'agent' not in m]
    if truly_missing:
        parts.append(f"[no data: {', '.join(truly_missing[:2])}{'...' if len(truly_missing) > 2 else ''}]")
    return '  '.join(parts) if parts else 'no context'


def _notes(r: Dict) -> str:
    """Context label plus a ⚠ flag when the classifier cross-check disagrees."""
    parts = [_ctx_label(r), f"hit {r.get('hit_rate', 0.5):.0%}"]
    src = r.get('hist_source')
    parts.append(f"dist:{'on-agent' if src == 'agent' else 'career'} ({r.get('n_maps', 0)}m)")
    if r.get('cross_note'):
        parts.append(f"⚠ {r['cross_note']}")
    return '  '.join(p for p in parts if p)


def _meta_label(meta: Optional[Dict], kind: str) -> str:
    """Render '<Name> (<metrics>)' from a loaded model's meta, e.g.
    'XGBoost (69.7% acc, AUC=0.760)' or 'LightGBM (R²=0.383, MAE=4.66)'."""
    if not meta:
        return 'Classifier' if kind == 'clf' else 'Regression'
    p = meta.get('perf', {})
    if kind == 'clf' and 'accuracy' in p:
        return f"{meta['name']} ({p['accuracy']:.1%} acc, AUC={p['auc']:.3f})"
    if kind == 'reg' and 'r2' in p:
        return f"{meta['name']} (R²={p['r2']:.3f}, MAE={p['mae']:.2f})"
    return meta.get('name', 'model')


def print_slate(rows: List[Dict], skipped_debuts: List, min_edge: float,
                min_appearances: int = 15, reg_meta: Optional[Dict] = None,
                clf_meta: Optional[Dict] = None,
                market_weight: float = kd.DEFAULT_MARKET_WEIGHT):
    today     = date.today().isoformat()
    using_clf = any(r.get('using_clf') for r in rows)

    # Separate: recommended bets vs explicit NO BETs vs weak-signal (below threshold)
    no_bets   = [r for r in rows if r.get('bet_rec') == 'NO BET']
    bettable  = [r for r in rows if r.get('bet_rec') != 'NO BET']
    bets      = [r for r in bettable if r['edge'] >= min_edge]
    weak      = [r for r in bettable if r['edge'] < min_edge]

    print()
    print(f'=== PrizePicks Valorant Bet Slate — {today} ===')
    print(f'{len(rows)} players with history  |  '
          f'{len(skipped_debuts)} skipped (debut/no data)')
    print(f'Mean model: {_meta_label(reg_meta, "reg")}  |  '
          f'cross-check: {_meta_label(clf_meta, "clf") if using_clf else "none"}')
    print(f'Recommend edge ≥ {min_edge:.0f}pt  |  Break-even at -110: {BREAKEVEN:.1%}  |  '
          f'stake = ¼-Kelly (cap 5%)')
    print(f'Market shrink λ={market_weight:.2f} (μ̂ pulled toward the line; '
          f'1=trust μ̂, 0=pure market)')
    print()

    # ── Recommended bets ────────────────────────────────────────────────────
    if not bets:
        print('  No bets meet the edge threshold today.')
    else:
        print(f'  ✔  RECOMMENDED BETS ({len(bets)})')
        hdr = (f"  {'#':>3}  {'Player':<20}  {'PP Line':>8}  {'P(win)':>7}  "
               f"{'Edge':>6}  {'Stake':>6}  {'Rec':<6}  Notes")
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for i, r in enumerate(bets, 1):
            print(f"  {i:>3}  {r['player']:<20}  {r['line']:>8.1f}  "
                  f"{r['p_win']:>6.1%}  {r['edge']:>5.1f}pt  {r['stake']:>5.1%}  "
                  f"{r['rec']:<6}  {_notes(r)}")

    # ── Explicit NO BETs ─────────────────────────────────────────────────────
    if no_bets:
        print()
        print(f'  ✖  NO BET — model flagged ({len(no_bets)})')
        for r in no_bets:
            prob_str = f"{r['prob_over']:.0%}" if r.get('prob_over') is not None else '—'
            print(f"  {r['player']:<20}  {r['line']:>8.1f}  P(OVER)={prob_str:>4}  "
                  f"{r['rec']:<6}  ← {r.get('filter_reason', '')}")

    # ── Weak signals ─────────────────────────────────────────────────────────
    if weak:
        print()
        print(f'  ~  Weak signal — edge < {min_edge:.0f}pt ({len(weak)})')
        for r in weak:
            print(f"       {r['player']:<20}  line {r['line']:.1f}  "
                  f"P(win)={r['p_win']:.1%}  edge {r['edge']:.1f}pt  {r['rec']}")

    # ── Debuts skipped ───────────────────────────────────────────────────────
    if skipped_debuts:
        print()
        print(f'  --  Skipped: debut / insufficient history (< {min_appearances} maps) --')
        for player, line, appearances in skipped_debuts:
            print(f"       {player:<20}  line {line:.1f}  appearances={appearances}")

    print()
    print(f'  MEAN MODEL : {_meta_label(reg_meta, "reg")}  (per-map μ̂)')
    if using_clf:
        print(f'  CROSS-CHECK: {_meta_label(clf_meta, "clf")}  (vetoes on confident disagreement)')
    print('  P(win)  = P(2-map kill SUM clears the line), from a mean-shifted')
    print('            bootstrap of the player\'s per-map history convolved over 2 maps.')
    print(f'  Edge    = P(win) − {BREAKEVEN:.1%} break-even, in percentage points.')
    print('  Stake   = quarter-Kelly fraction of bankroll at -110 (capped 5%).')
    print('  hit     = % of career maps over the per-map line (reference only).')
    print('  ~ prefix = map/agent guessed from history, not confirmed.')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'player_cache.pkl')


def get_player_cache(limit: Optional[int], rebuild: bool = False,
                     path: str = CACHE_PATH) -> 'PlayerCache':
    """Load a prebuilt player cache if present (≈2s), else build it from the
    JSON match history (≈minutes) and save it for next time.

    The cache is just aggregated lookup tables, so it only needs rebuilding when
    new matches are scraped — pass --rebuild-cache then (or delete the pickle).
    """
    if not rebuild and os.path.exists(path):
        # Warn if the match DB is newer than the cached snapshot.
        try:
            db = os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db')
            if os.path.exists(db) and os.path.getmtime(db) > os.path.getmtime(path):
                print('  Note: match DB is newer than the cache — consider --rebuild-cache')
        except OSError:
            pass
        print(f'Loading prebuilt player cache ({os.path.basename(path)})...')
        cache = PlayerCache.from_file(path)
        print(f'Cache loaded: {len(cache.player_kills)} players, '
              f'{len(cache.db_kill_hist)} DB kill distributions')
        return cache

    print('Building player cache from match history (one-time — reused after this)...')
    cache = PlayerCache(limit=limit)
    cache.save(path)
    print(f'Cache saved → {os.path.basename(path)}.  Future runs load it in ~2s.')
    return cache


def main():
    parser = argparse.ArgumentParser(description='Generate daily PrizePicks Valorant bet slate')
    parser.add_argument('--min-edge',      type=float, default=3.0,
                        help='Minimum edge in percentage POINTS over the 52.4%% break-even '
                             'to recommend a bet (e.g. 3 = need p_win >= 55.4%%; default: 3)')
    parser.add_argument('--context',       type=str,   default=None,
                        help='JSON file with per-player matchup context (team, opponent, maps, agent)')
    parser.add_argument('--min-appearances', type=int, default=15,
                        help='Skip players with fewer than N map appearances in history (default: 15)')
    parser.add_argument('--market-weight', type=float, default=kd.DEFAULT_MARKET_WEIGHT,
                        help='λ: fraction of our μ̂-vs-line disagreement to keep (0=pure market, '
                             f'1=trust μ̂ fully). Shrinks a hot μ̂ toward the line. Default: {kd.DEFAULT_MARKET_WEIGHT}')
    parser.add_argument('--cache-matches', type=int,   default=None,
                        help='Cap match files when BUILDING the cache (default: all). '
                             'Ignored once a prebuilt cache exists.')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Rebuild the player cache from match history (do this after scraping)')
    parser.add_argument('--league-id',     type=int,   default=159,
                        help='PrizePicks league ID for Valorant (default: 159)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not auto-save picks to bet_results.csv')
    args = parser.parse_args()

    # Load context file
    context: Dict[str, Dict] = {}
    if args.context:
        with open(args.context) as f:
            raw = json.load(f)
        # normalise keys to lowercase to match PrizePicks name format
        context = {k.lower(): v for k, v in raw.items()}
        print(f'Loaded context for {len(context)} players from {args.context}')

    # Load regression model
    print('Loading regression model...')
    model, scaler, feature_cols, reg_meta = load_gbr()
    _perf = reg_meta['perf']
    _reg_metrics = (f"R²={_perf['r2']:.3f}, MAE={_perf['mae']:.2f}"
                    if 'r2' in _perf else 'metrics n/a')
    print(f"{reg_meta['name']} regression loaded ({_reg_metrics}, {len(feature_cols)} features)")

    # Load classifier (OVER/UNDER) — optional, falls back to regression if missing
    clf_result = load_classifier()
    if clf_result is not None:
        clf, clf_scaler, clf_cols, clf_meta = clf_result
        _cperf = clf_meta['perf']
        _clf_metrics = (f"{_cperf['accuracy']:.1%} acc, AUC={_cperf['auc']:.3f}"
                        if 'accuracy' in _cperf else 'metrics n/a')
        print(f"{clf_meta['name']} classifier loaded ({_clf_metrics}, {len(clf_cols)} features)")
    else:
        clf = clf_scaler = clf_cols = None
        print('Classifier not found — using regression for edge calculation')

    # Build player cache (or load the prebuilt one — the big speedup)
    cache = get_player_cache(args.cache_matches, rebuild=args.rebuild_cache)

    # Fetch live lines
    print('Fetching PrizePicks lines...')
    client = PrizePicksClient(league_id=args.league_id)
    lines  = client.fetch_kill_lines()

    if not lines:
        print()
        print('No active Valorant kill lines found on PrizePicks.')
        print('Possible reasons:')
        print('  - No Valorant matches scheduled today')
        print('  - League ID has changed — try --league-id with a different value')
        print('  - PrizePicks API is temporarily unavailable')
        return

    print(f'Found {len(lines)} active kill lines')

    # Build and print slate
    rows, skipped_debuts = build_slate(
        cache, model, scaler, feature_cols, lines, context,
        args.min_edge, args.min_appearances,
        clf=clf, clf_scaler=clf_scaler, clf_cols=clf_cols,
        market_weight=args.market_weight,
    )
    print_slate(rows, skipped_debuts, args.min_edge, args.min_appearances,
                reg_meta=reg_meta, clf_meta=clf_meta if clf_result is not None else None,
                market_weight=args.market_weight)

    # Auto-save recommended bets to results tracker
    if not args.no_save:
        # Log the FULL market first — every line we saw, bet or not — so the
        # corpus is complete for backtesting / market-aware retraining.
        n_logged = _log_line_history(rows)
        if n_logged:
            print(f'  Logged {n_logged} line(s) to line_history.csv')

        bets = [r for r in rows if r.get('bet_rec') != 'NO BET' and r['edge'] >= args.min_edge]
        if bets:
            n_saved = _save_slate(bets)
            if n_saved:
                print(f'  Saved {n_saved} pick(s) to bet_results.csv')
                print('  Enter results later:  python results_tracker.py result <player> <actual>')
                print()


if __name__ == '__main__':
    main()
