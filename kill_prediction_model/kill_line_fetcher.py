#!/usr/bin/env python3
"""
Kill line fetcher — real and high-quality synthetic kill lines.

Two backends:
  SmartSyntheticLine  — weighted blend of player signals + opponent adjustment
  PrizePicksClient    — fetches live lines from PrizePicks public API (no auth)

KillLineFetcher orchestrates: tries live first, falls back to synthetic.

Usage (live):
  python kill_line_fetcher.py aspas
  python kill_line_fetcher.py          # prints first 20 active lines
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AVG_ROUNDS = 18.5


# ---------------------------------------------------------------------------
# Smart synthetic line (for backtesting / offline fallback)
# ---------------------------------------------------------------------------

class SmartSyntheticLine:
    """
    Weighted ensemble that better approximates market kill lines than a plain
    rolling average.

    Weight scheme:
      50 % recent form      (rolling 10-match average)
      30 % map history      (player's avg kills on these maps)
      20 % career rate      (KPR × avg rounds/map)

    Plus an opponent-difficulty adjustment: tough opponents lower the line by
    up to 1.5 kills (mirrors how books shade lines for elite defensive squads).
    Weights re-normalize automatically when a signal is missing.
    """

    def __init__(self, avg_rounds: float = AVG_ROUNDS):
        self.avg_rounds = avg_rounds

    def compute(self, row: pd.Series) -> float:
        signals, weights = [], []

        recent = float(row.get('recent_avg_kills') or 0)
        if recent > 0:
            signals.append(recent)
            weights.append(0.5)

        map_avg = float(row.get('player_map_avg_kills') or 0)
        if map_avg > 0:
            signals.append(map_avg)
            weights.append(0.3)

        kpr = float(row.get('db_kills_per_round') or 0)
        if kpr > 0:
            signals.append(kpr * self.avg_rounds)
            weights.append(0.2)

        if not signals:
            return 13.0

        total = sum(weights)
        line = sum(s * w / total for s, w in zip(signals, weights))

        # Opponent-difficulty adjustment
        opp_str = float(row.get('opponent_team_strength') or 1.0)
        opp_adj = float(np.clip((opp_str - 1.0) * -0.5, -1.5, 1.5))
        line += opp_adj

        return float(np.clip(line, 5.0, 35.0))


# ---------------------------------------------------------------------------
# PrizePicks live line client
# ---------------------------------------------------------------------------

@dataclass
class LiveLine:
    player_name: str
    stat_type: str
    line_score: float
    source: str = 'prizepicks'


class PrizePicksClient:
    """
    Fetches live Valorant player prop lines from PrizePicks' public API.

    PrizePicks' web/mobile app uses this endpoint; no authentication is
    required.  The league_id for Valorant may change over time — if this
    stops returning results, inspect network traffic in the PrizePicks app
    and update DEFAULT_LEAGUE_ID below.

    Known league IDs (mid-2025):
      Valorant ≈ 36
    """

    BASE_URL = 'https://api.prizepicks.com'
    DEFAULT_LEAGUE_ID = 36
    KILL_STAT_TYPES = {'Kills', 'Kills+Assists'}

    def __init__(self, league_id: int = DEFAULT_LEAGUE_ID, timeout: int = 10):
        self.league_id = league_id
        self.timeout = timeout
        self._cache: Optional[Dict[str, float]] = None

    def _get(self, path: str) -> dict:
        url = f'{self.BASE_URL}{path}'
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ValorantPredictor/1.0)',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def fetch_kill_lines(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Returns {normalized_player_name: kill_line} for all active Valorant
        kill projections.  Results are cached per instance.

        Returns empty dict if the API is unreachable or returns no results.
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        try:
            data = self._get(
                f'/projections?league_id={self.league_id}'
                '&per_page=250&single_stat=true'
            )
        except Exception as e:
            logger.warning(f'PrizePicks fetch failed: {e}')
            return {}

        lines: Dict[str, float] = {}
        for item in data.get('data', []):
            attrs = item.get('attributes', {})
            stat = attrs.get('stat_type', '')
            if stat not in self.KILL_STAT_TYPES:
                continue
            name = attrs.get('name', '').strip().lower()
            score_raw = attrs.get('line_score')
            if not name or score_raw is None:
                continue
            try:
                lines[name] = float(score_raw)
            except (ValueError, TypeError):
                continue

        logger.info(f'Fetched {len(lines)} PrizePicks kill lines (league {self.league_id})')
        self._cache = lines
        return lines

    def get_kill_line(self, player_name: str) -> Optional[float]:
        """Return kill line for player (case-insensitive), or None if not found."""
        lines = self.fetch_kill_lines()
        return lines.get(player_name.strip().lower())

    def list_available_players(self) -> list[str]:
        """Return all players with active kill lines, sorted."""
        return sorted(self.fetch_kill_lines().keys())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class KillLineFetcher:
    """
    Unified kill line source.

      get_live_line(player_name)       → live PrizePicks lookup (or None)
      compute_synthetic(row)           → smart blend from feature row
      get_line(player_name, row=None)  → live with synthetic fallback
    """

    def __init__(self, league_id: int = PrizePicksClient.DEFAULT_LEAGUE_ID):
        self._pp = PrizePicksClient(league_id=league_id)
        self._synthetic = SmartSyntheticLine()

    def get_live_line(self, player_name: str) -> Optional[float]:
        return self._pp.get_kill_line(player_name)

    def compute_synthetic(self, row: pd.Series) -> float:
        return self._synthetic.compute(row)

    def get_line(
        self,
        player_name: str,
        row: Optional[pd.Series] = None,
    ) -> tuple[float, str]:
        """
        Returns (kill_line, source).
        source is one of: 'prizepicks', 'synthetic', 'fallback'.
        """
        live = self.get_live_line(player_name)
        if live is not None:
            return live, 'prizepicks'
        if row is not None:
            return self.compute_synthetic(row), 'synthetic'
        return 13.0, 'fallback'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    player_arg = sys.argv[1] if len(sys.argv) > 1 else None
    client = PrizePicksClient()
    lines = client.fetch_kill_lines()

    if not lines:
        print('No live lines returned.  Possible reasons:')
        print('  • No active Valorant matches on PrizePicks today')
        print('  • league_id has changed — update DEFAULT_LEAGUE_ID in kill_line_fetcher.py')
        print('  • API is temporarily unavailable')
        sys.exit(0)

    if player_arg:
        key = player_arg.strip().lower()
        val = lines.get(key)
        if val is not None:
            print(f'{player_arg}: {val} kills')
        else:
            print(f'{player_arg!r} not found in active lines.')
            print(f'Available players: {", ".join(sorted(lines.keys())[:10])} ...')
    else:
        print(f'Active Valorant kill lines ({len(lines)} players):')
        for name, line in sorted(lines.items()):
            print(f'  {name}: {line}')
