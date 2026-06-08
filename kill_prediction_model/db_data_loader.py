#!/usr/bin/env python3
"""
DB Data Loader — reads training data directly from valorant_matches.db.

Much faster than the JSON loader (single SQL query vs 52k file reads).
Produces the same (X, y) tuple as EnhancedDataLoader.prepare_training_data().

Limitations vs JSON loader:
  - No agent data  → agent features fall back to UNKNOWN_ROLE
  - No vlr.gg rating → uses kdr as proxy for match_rating

Usage:
  from db_data_loader import DBDataLoader
  loader = DBDataLoader()
  X, y = loader.prepare_training_data()
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scraper'))

from enhanced_data_loader import AGENT_ROLES, UNKNOWN_ROLE
from db_utils import get_connection as get_career_conn

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db')

FEATURE_COLUMNS = [
    'db_rating', 'db_average_combat_score', 'db_kill_deaths',
    'db_kills_per_round', 'db_assists_per_round',
    'db_first_kills_per_round', 'db_first_deaths_per_round',
    'team_strength',
    'opponent_team_strength',
    'opponent_kills_allowed_per_map',
    'recent_avg_kills', 'recent_avg_rating',
    'recent_avg_kills_3',        # rolling avg of last 3 maps (faster recency)
    'form_slope',
    'days_since_last_match',     # rest days — freshness / preparation
    'h2h_avg_kills',
    'h2h_data_exists',
    'player_map_avg_kills',
    'avg_rounds_vs_opponent',    # expected rounds based on team-matchup history
    'kill_std',
    'agent_role_ordinal', 'is_duelist', 'player_agent_avg_kills',
]


# ---------------------------------------------------------------------------
# SQL query
# ---------------------------------------------------------------------------

_QUERY = """
SELECT
    p.name            AS player_name,
    t.name            AS team,
    ot.name           AS opponent_team,
    mn.map_name,
    m.match_date,
    m.id              AS match_id,
    pms.kills         AS match_kills,
    pms.deaths        AS match_deaths,
    pms.assists       AS match_assists,
    pms.acs           AS match_acs,
    pms.adr           AS match_adr,
    -- kdr column is corrupt (stores kd_diff for ~50% of rows); compute directly
    CAST(pms.kills AS REAL) / NULLIF(pms.deaths, 0) AS match_rating
FROM player_match_stats pms
JOIN players p  ON p.id  = pms.player_id
JOIN teams   t  ON t.id  = pms.team_id
JOIN maps    mn ON mn.id = pms.map_id
JOIN matches m  ON m.id  = pms.match_id
JOIN teams   ot ON ot.id = CASE
    WHEN m.team1_id = pms.team_id THEN m.team2_id
    ELSE m.team1_id
END
WHERE pms.kills BETWEEN 1 AND 40
  AND pms.acs    > 0
  AND pms.adr    > 0
  AND pms.deaths > 0
  AND mn.map_name != ''
"""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DBDataLoader:
    """
    Loads training data from valorant_matches.db with a single SQL query.
    Returns the same (X, y) interface as EnhancedDataLoader.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._career_cache = None

    # -- career stats from vlr_players.db ------------------------------------

    def _load_career_stats(self) -> pd.DataFrame:
        if self._career_cache is not None:
            return self._career_cache
        try:
            conn = get_career_conn()
            df = pd.read_sql_query("""
                SELECT name, rating, average_combat_score, kill_deaths,
                       kills_per_round, assists_per_round,
                       first_kills_per_round, first_deaths_per_round
                FROM players
            """, conn)
            conn.close()
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Deduplicate: same player may appear under multiple teams.
            # Average numeric cols; keep the canonical (highest-ACS) name spelling.
            df['_name_lower'] = df['name'].str.lower()
            numeric_cols = ['rating', 'average_combat_score', 'kill_deaths',
                            'kills_per_round', 'assists_per_round',
                            'first_kills_per_round', 'first_deaths_per_round']
            agg = df.groupby('_name_lower')[numeric_cols].mean().reset_index()
            # Re-attach original name spelling (pick highest ACS entry)
            best_name = (df.sort_values('average_combat_score', ascending=False)
                           .drop_duplicates('_name_lower')[['_name_lower', 'name']])
            agg = agg.merge(best_name, on='_name_lower').drop(columns=['_name_lower'])

            self._career_cache = agg
            return agg
        except Exception as e:
            print(f'  Warning: could not load career stats: {e}')
            return pd.DataFrame()

    # -- raw data from matches DB --------------------------------------------

    def _load_raw(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query(_QUERY, conn)
        conn.close()
        return df

    # -- feature engineering -------------------------------------------------

    def _add_career_features(self, df: pd.DataFrame) -> pd.DataFrame:
        career = self._load_career_stats()
        if career.empty:
            for col in ['db_rating', 'db_average_combat_score', 'db_kill_deaths',
                        'db_kills_per_round', 'db_assists_per_round',
                        'db_first_kills_per_round', 'db_first_deaths_per_round']:
                df[col] = 0.0
            return df

        career = career.rename(columns={
            'rating':                  'db_rating',
            'average_combat_score':    'db_average_combat_score',
            'kill_deaths':             'db_kill_deaths',
            'kills_per_round':         'db_kills_per_round',
            'assists_per_round':       'db_assists_per_round',
            'first_kills_per_round':   'db_first_kills_per_round',
            'first_deaths_per_round':  'db_first_deaths_per_round',
        })
        df = df.merge(career[['name'] + [c for c in career.columns if c.startswith('db_')]],
                      left_on='player_name', right_on='name', how='left')
        df = df.drop(columns=['name'], errors='ignore')
        for col in [c for c in df.columns if c.startswith('db_')]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # team_strength = own team avg rating in this match
        team_avg = (df.groupby(['match_id', 'team'])['match_rating']
                    .mean().reset_index()
                    .rename(columns={'match_rating': 'team_strength'}))
        df = df.merge(team_avg, on=['match_id', 'team'], how='left')

        # opponent_team_strength = ACTUAL opposing team avg rating in this match
        opp_avg = (df.groupby(['match_id', 'team'])['match_rating']
                   .mean().reset_index()
                   .rename(columns={'team': 'opponent_team',
                                    'match_rating': 'opponent_team_strength'}))
        df = df.merge(opp_avg, on=['match_id', 'opponent_team'], how='left')
        df['opponent_team_strength'] = df['opponent_team_strength'].fillna(df['team_strength'])

        # opponent_kills_allowed_per_map = how many kills opponent concedes on average
        opp_allowed = (df.groupby('opponent_team')['match_kills']
                       .mean().reset_index()
                       .rename(columns={'match_kills': 'opponent_kills_allowed_per_map'}))
        df = df.merge(opp_allowed, on='opponent_team', how='left')
        df['opponent_kills_allowed_per_map'] = df['opponent_kills_allowed_per_map'].fillna(
            df['match_kills'].mean()
        )
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        df = df.sort_values(['player_name', 'match_date']).reset_index(drop=True)

        grp = df.groupby('player_name', group_keys=False)

        df['recent_avg_kills'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=2).mean()
        )
        df['recent_avg_rating'] = grp['match_rating'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=2).mean()
        )
        df['recent_avg_kills_3'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(df['match_kills'].mean())
        df['form_slope'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda y: float(np.polyfit(range(len(y)), y, 1)[0]),
                raw=True,
            )
        )
        df['form_slope'] = df['form_slope'].fillna(0.0)

        # days_since_last_match — rest days between consecutive map appearances
        df['_prev_date'] = df.groupby('player_name')['match_date'].shift(1)
        df['days_since_last_match'] = (
            (df['match_date'] - df['_prev_date']).dt.days
            .fillna(7.0).clip(0, 30)
        )
        df = df.drop(columns=['_prev_date'])

        # player_map_avg_kills (leave-one-out)
        grp_map   = df.groupby(['player_name', 'map_name'])['match_kills']
        map_sum   = grp_map.transform('sum')
        map_count = grp_map.transform('count')
        player_overall = df.groupby('player_name')['match_kills'].transform('mean')
        df['player_map_avg_kills'] = ((map_sum - df['match_kills']) / (map_count - 1)
                                      ).fillna(player_overall)

        # h2h_avg_kills (leave-one-out)
        grp_h2h   = df.groupby(['player_name', 'opponent_team'])['match_kills']
        h2h_sum   = grp_h2h.transform('sum')
        h2h_count = grp_h2h.transform('count')
        df['h2h_avg_kills'] = ((h2h_sum - df['match_kills']) / (h2h_count - 1)
                               ).fillna(player_overall)

        # h2h_data_exists — 1 if real h2h data, 0 if first-time matchup
        df['h2h_data_exists'] = (h2h_count > 1).astype(float)

        # kill_std — player's historical kill standard deviation
        df['kill_std'] = (
            df.groupby('player_name')['match_kills'].transform('std').fillna(3.0)
        )

        return df

    def _add_rounds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Estimate rounds per map from total kills (10 players ≈ 6 kills/round in pro play)
        map_totals = (df.groupby(['match_id', 'map_name'])['match_kills']
                      .sum().reset_index()
                      .rename(columns={'match_kills': 'map_total_kills'}))
        map_totals['estimated_rounds'] = map_totals['map_total_kills'] / 6.0
        df = df.merge(map_totals[['match_id', 'map_name', 'estimated_rounds']],
                      on=['match_id', 'map_name'], how='left')

        # avg_rounds_vs_opponent — leave-one-out avg for this team-opponent pairing
        grp_r = df.groupby(['team', 'opponent_team'])['estimated_rounds']
        r_sum   = grp_r.transform('sum')
        r_count = grp_r.transform('count')
        global_avg = df['estimated_rounds'].mean()
        df['avg_rounds_vs_opponent'] = (
            (r_sum - df['estimated_rounds']) / (r_count - 1)
        ).fillna(global_avg)

        return df

    def _add_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # No agent in DB — fall back to UNKNOWN_ROLE for all rows
        df['agent_role_ordinal']    = UNKNOWN_ROLE
        df['is_duelist']            = 0.0
        df['player_agent_avg_kills'] = (
            df.groupby('player_name')['match_kills'].transform('mean')
        )
        return df

    # -- public API ----------------------------------------------------------

    def prepare_training_data(self) -> tuple:
        print('=== Preparing Training Data from DB ===')
        print(f'  Loading from {self.db_path} ...')

        df = self._load_raw()
        print(f'  {len(df):,} raw rows loaded')

        df = self._add_career_features(df)
        df = self._add_context_features(df)
        df = self._add_rolling_features(df)
        df = self._add_rounds_features(df)
        df = self._add_agent_features(df)

        # Keep only rows with career stats and positive kills
        df_clean = df[
            (df['db_rating'] > 0) &
            (df['match_kills'] > 0)
        ].copy()

        numeric_cols = FEATURE_COLUMNS + ['match_kills']
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].median(numeric_only=True)
        )
        df_clean = df_clean.dropna(subset=numeric_cols)

        if df_clean.empty:
            raise RuntimeError('No usable training rows after filtering.')

        X = df_clean[FEATURE_COLUMNS].copy()
        X['player_name'] = df_clean['player_name'].values
        y = df_clean['match_kills']

        print(f'  Final dataset: {len(X):,} rows, {len(X.columns)} features')
        print(f'  Target range [{y.min():.0f}, {y.max():.0f}]  mean={y.mean():.2f}')
        return X, y
