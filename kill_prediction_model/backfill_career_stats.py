#!/usr/bin/env python3
"""
Backfill career stats for players missing from vlr_players.db.

Root causes identified:
  1. vlr_players.db and valorant_matches.db are populated by independent scrapers
     with no sync — 1,937 players with ≥15 maps have no career entries at all.
  2. The `kdr` column in player_match_stats is corrupt: database_schema.py stored
     kd_diff (kills−deaths) into it with fallback to actual KD ratio, so ~50% of
     rows contain negative values. We skip kdr and compute KD from kills/deaths directly.

This script computes approximate career stats from match history and inserts
them for any player NOT already present in vlr_players.db (by lowercase name).
Existing real scraped data is never overwritten (INSERT OR IGNORE per name).

Approximations:
  db_rating              ≈ 0.002382 * avg_acs + 0.4872  (r=0.516 vs real VLR rating)
  db_average_combat_score = avg(pms.acs)
  db_kill_deaths         = avg(kills/deaths)  computed directly, not from kdr column
  db_kills_per_round     = avg(kills / est_rounds)  est_rounds = total_map_kills/6
  db_assists_per_round   = avg(assists / est_rounds)
  db_first_kills_per_round  = 0.092  (league mean — fk column not per-round normalized)
  db_first_deaths_per_round = 0.107  (league mean)

Usage:
  python backfill_career_stats.py              # backfill all missing players
  python backfill_career_stats.py --dry-run    # preview without writing
  python backfill_career_stats.py --min-maps 5 # lower threshold
"""

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
from db_utils import get_connection as career_conn

MATCHES_DB = os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db')

_ACS_TO_RATING_SLOPE     = 0.002382
_ACS_TO_RATING_INTERCEPT = 0.4872
_LEAGUE_FKPR = 0.092
_LEAGUE_FDPR = 0.107
MIN_MAPS_DEFAULT = 10

_UNKNOWN_TEAM_PATTERNS = ['unknown', 'unaffiliated', 'tbd', 'free agent']


def _is_real_team(name: str) -> bool:
    n = (name or '').lower().strip()
    return bool(n) and not any(p in n for p in _UNKNOWN_TEAM_PATTERNS)


def load_match_data() -> pd.DataFrame:
    conn = sqlite3.connect(MATCHES_DB)
    df = pd.read_sql_query("""
        SELECT
            LOWER(p.name)  AS player_name,
            p.name         AS raw_name,
            t.name         AS team,
            pms.kills,
            pms.deaths,
            pms.assists,
            pms.acs,
            pms.adr,
            m.match_date,
            pms.match_id,
            pms.map_id
        FROM player_match_stats pms
        JOIN players p ON p.id = pms.player_id
        JOIN teams   t ON t.id = pms.team_id
        JOIN matches m ON m.id = pms.match_id
        WHERE pms.kills  BETWEEN 1 AND 40
          AND pms.acs    > 0
          AND pms.deaths > 0
    """, conn)
    conn.close()
    return df


def compute_career_stats(df: pd.DataFrame, min_maps: int) -> pd.DataFrame:
    # Compute KD directly — don't touch the corrupt kdr column
    df = df.copy()
    df['kd'] = df['kills'] / df['deaths'].clip(lower=1)

    # Estimated rounds per map from total kills (10 players, ~6 kills/round in pro play)
    map_totals = (df.groupby(['match_id', 'map_id'])['kills']
                  .sum().reset_index()
                  .rename(columns={'kills': 'map_total_kills'}))
    map_totals['est_rounds'] = (map_totals['map_total_kills'] / 6.0).clip(lower=5)
    df = df.merge(map_totals[['match_id', 'map_id', 'est_rounds']],
                  on=['match_id', 'map_id'], how='left')
    df['est_rounds'] = df['est_rounds'].fillna(13.0)
    df['kpr'] = df['kills']   / df['est_rounds']
    df['apr'] = df['assists'] / df['est_rounds']

    # Most recent real team — sort by date, take last non-unknown team per player
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df_sorted = df.sort_values('match_date')

    def latest_real_team(group):
        real = group[group['team'].apply(_is_real_team)]
        if not real.empty:
            return real.iloc[-1]['team']
        return group.iloc[-1]['team']

    teams = (df_sorted.groupby('player_name')
             .apply(lambda g: pd.Series({'last_real_team': latest_real_team(g)}),
                    include_groups=False)
             .reset_index())
    teams.columns = ['player_name', 'last_real_team']

    # Aggregate stats per player
    agg = df.groupby('player_name').agg(
        raw_name  = ('raw_name', 'first'),
        maps      = ('kills',    'count'),
        avg_acs   = ('acs',      'mean'),
        avg_kd    = ('kd',       'mean'),
        avg_kpr   = ('kpr',      'mean'),
        avg_apr   = ('apr',      'mean'),
    ).reset_index()

    agg = agg.merge(teams, on='player_name')
    agg = agg[agg['maps'] >= min_maps].copy()

    agg['rating'] = (_ACS_TO_RATING_SLOPE * agg['avg_acs'] + _ACS_TO_RATING_INTERCEPT).round(3)
    agg['rating']  = agg['rating'].clip(0.30, 1.80)
    agg['avg_acs'] = agg['avg_acs'].round(1)
    agg['avg_kd']  = agg['avg_kd'].round(3)
    agg['avg_kpr'] = agg['avg_kpr'].round(3)
    agg['avg_apr'] = agg['avg_apr'].round(3)

    return agg


def get_existing_career_names() -> set:
    conn = career_conn()
    cur  = conn.cursor()
    cur.execute('SELECT LOWER(name) FROM players')
    names = {r[0] for r in cur.fetchall()}
    conn.close()
    return names


def backfill(stats: pd.DataFrame, existing_names: set, dry_run: bool) -> int:
    missing = stats[~stats['player_name'].isin(existing_names)].copy()
    print(f'\nPlayers to backfill: {len(missing):,}  '
          f'(already in career DB: {len(stats) - len(missing):,})')

    if missing.empty:
        print('Nothing to insert.')
        return 0

    if dry_run:
        print('\n[DRY RUN — top 20 that would be inserted]')
        print(missing.sort_values('maps', ascending=False)
              [['raw_name', 'maps', 'avg_acs', 'avg_kd', 'rating', 'last_real_team']]
              .head(20).to_string(index=False))
        return len(missing)

    conn = career_conn()
    cur  = conn.cursor()
    inserted = 0
    skipped  = 0

    for _, row in missing.iterrows():
        try:
            cur.execute("""
                INSERT OR IGNORE INTO players (
                    name, team,
                    rating, average_combat_score, kill_deaths,
                    kills_per_round, assists_per_round,
                    first_kills_per_round, first_deaths_per_round,
                    kill_assists_survived_traded,
                    average_damage_per_round,
                    headshot_percentage,
                    clutch_success_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
            """, (
                row['raw_name'],
                row['last_real_team'],
                str(row['rating']),
                str(row['avg_acs']),
                str(row['avg_kd']),
                str(row['avg_kpr']),
                str(row['avg_apr']),
                str(_LEAGUE_FKPR),
                str(_LEAGUE_FDPR),
            ))
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f'  [WARN] {row["raw_name"]}: {e}')

    conn.commit()
    conn.close()
    print(f'Inserted: {inserted:,}  |  Skipped (name+team conflict): {skipped:,}')
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Backfill career stats from match data')
    parser.add_argument('--min-maps', type=int, default=MIN_MAPS_DEFAULT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('=== Career Stats Backfill ===')
    print(f'Min maps: {args.min_maps}')

    print('\n[1] Loading match data ...')
    df = load_match_data()
    print(f'    {len(df):,} rows  |  {df["player_name"].nunique():,} unique players')

    print('\n[2] Computing career aggregates ...')
    stats = compute_career_stats(df, min_maps=args.min_maps)
    print(f'    {len(stats):,} players with >= {args.min_maps} maps')
    print(f'    KD range: [{stats["avg_kd"].min():.2f}, {stats["avg_kd"].max():.2f}]  '
          f'mean={stats["avg_kd"].mean():.2f}')

    print('\n[3] Loading existing career DB ...')
    existing = get_existing_career_names()
    print(f'    {len(existing):,} names in vlr_players.db')

    print('\n[4] Backfilling ...')
    n = backfill(stats, existing, dry_run=args.dry_run)

    if not args.dry_run and n > 0:
        conn = career_conn()
        cur  = conn.cursor()
        cur.execute("SELECT name, team, rating, average_combat_score, kill_deaths "
                    "FROM players WHERE LOWER(name) = 'keiko'")
        row = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM players")
        total = cur.fetchone()[0]
        conn.close()
        print(f'\nTotal entries in vlr_players.db: {total:,}')
        if row:
            print(f'Keiko: name={row[0]}, team={row[1]}, '
                  f'rating={row[2]}, acs={row[3]}, kd={row[4]}')
        else:
            print('WARN: keiko still not found — check spelling')


if __name__ == '__main__':
    main()
