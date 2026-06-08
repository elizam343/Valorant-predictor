#!/usr/bin/env python3
"""
Player name alias resolver.

Fetches live PrizePicks names, cross-references against every player name
in valorant_matches.db, and writes name_aliases.json with fuzzy-match
suggestions for you to review and correct.

Usage:
  python name_resolver.py                      # check today's PP names
  python name_resolver.py --min-appearances 5  # flag anyone with < 5 map appearances
  python name_resolver.py --all                # check ALL known PP names, not just today's

After running, open name_aliases.json and verify / correct the mappings.
bet_slate.py loads this file automatically.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scraper'))

from kill_line_fetcher import PrizePicksClient

DB_PATH      = Path(__file__).parent.parent / 'Scraper' / 'valorant_matches.db'
ALIASES_FILE = Path(__file__).parent / 'name_aliases.json'

# ---------------------------------------------------------------------------
# Leet / common gaming substitutions (applied symmetrically)
# ---------------------------------------------------------------------------
_SUBS = str.maketrans('013456789', 'oieashbpg')


def _normalize(name: str) -> str:
    """Lowercase + strip + apply common leet substitutions."""
    return name.lower().strip().translate(_SUBS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(pp: str, vlr: str) -> float:
    """
    Return similarity score 0-1 between a PP name and a VLR name.
    Combines three signals:
      - SequenceMatcher ratio on normalized strings (main signal)
      - Bonus if one is a substring of the other
      - Bonus for shared prefix of length ≥ 3
    """
    pp_n  = _normalize(pp)
    vlr_n = _normalize(vlr)

    base = SequenceMatcher(None, pp_n, vlr_n).ratio()

    # substring bonus
    if pp_n in vlr_n or vlr_n in pp_n:
        base = min(1.0, base + 0.15)

    # shared prefix bonus (length ≥ 3)
    prefix = 0
    for a, b in zip(pp_n, vlr_n):
        if a == b:
            prefix += 1
        else:
            break
    if prefix >= 3:
        base = min(1.0, base + 0.05 * min(prefix, 4))

    return round(base, 4)


def find_candidates(pp_name: str, vlr_names: list[str], top_n: int = 5) -> list[tuple]:
    """Return [(vlr_name, score), ...] sorted by score descending."""
    scored = [(vlr, _score(pp_name, vlr)) for vlr in vlr_names]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_vlr_names() -> list[str]:
    """All distinct player names from the matches DB."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute('SELECT DISTINCT name FROM players WHERE name IS NOT NULL AND name != ""')
    names = [r[0] for r in cur.fetchall()]
    conn.close()
    return names


def load_appearances_from_db() -> dict[str, int]:
    """
    Count how many map appearances each player has in player_match_stats.
    Keyed by lowercase name.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT LOWER(p.name), COUNT(*)
        FROM player_match_stats pms
        JOIN players p ON p.id = pms.player_id
        WHERE pms.kills BETWEEN 1 AND 40 AND pms.acs > 0
        GROUP BY LOWER(p.name)
    """)
    result = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Alias file helpers
# ---------------------------------------------------------------------------

def load_aliases() -> dict[str, str]:
    if not ALIASES_FILE.exists():
        return {}
    data = json.loads(ALIASES_FILE.read_text())
    return {k: v for k, v in data.items() if not k.startswith('_')}


def save_aliases(aliases: dict, suggestions: dict) -> None:
    """
    Write name_aliases.json.
    Format:
      {
        "_note": "...",
        "_suggestions": { pp_name: [{vlr_name, score}, ...] },
        pp_name: "vlr_canonical_name",   ← edit these
        ...
      }
    """
    out = {
        '_note': (
            'Maps PrizePicks player names (lowercase) to their VLR.gg canonical names. '
            'Edit the values on the right to correct wrong suggestions. '
            'Delete a line entirely to leave that name unmapped (uses PP name as-is).'
        ),
        '_suggestions': suggestions,
    }
    out.update(aliases)
    ALIASES_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve(pp_names: list[str], min_appearances: int, vlr_names: list[str],
            appearances: dict[str, int], existing_aliases: dict[str, str]) -> tuple[dict, dict]:
    """
    For each PP name, decide whether it needs an alias entry.
    Returns (new_aliases, suggestions).
    """
    new_aliases  = dict(existing_aliases)
    suggestions  = {}

    vlr_lower_to_canon = {}
    for n in vlr_names:
        vlr_lower_to_canon[n.lower()] = n

    for pp in sorted(pp_names):
        pp_l = pp.lower()

        # Already aliased — skip unless appearances are still low
        if pp_l in existing_aliases:
            target = existing_aliases[pp_l].lower()
            app    = appearances.get(target, 0)
            status = f'aliased → {existing_aliases[pp_l]} ({app} appearances)'
            print(f'  {pp:<22} {status}')
            continue

        # Check direct lowercase match
        app = appearances.get(pp_l, 0)

        if app >= min_appearances:
            # Sufficient data — no alias needed
            canon = vlr_lower_to_canon.get(pp_l, pp)
            print(f'  {pp:<22} OK  {app} appearances  ({canon})')
            continue

        # Low / zero appearances — fuzzy match
        candidates = find_candidates(pp, vlr_names, top_n=5)
        top_name, top_score = candidates[0] if candidates else ('', 0.0)
        top_app = appearances.get(top_name.lower(), 0)

        suggestions[pp_l] = [
            {'vlr_name': c, 'score': s, 'appearances': appearances.get(c.lower(), 0)}
            for c, s in candidates
        ]

        confidence = '  HIGH' if top_score >= 0.85 else ' MED' if top_score >= 0.65 else '  LOW'
        print(f'  {pp:<22} {app:>4} appearances →'
              f' best match: {top_name!r:30} score={top_score:.2f}{confidence}'
              f'  ({top_app} map apps)')

        # Auto-accept high-confidence matches where target has more data
        if top_score >= 0.90 and top_app > app:
            new_aliases[pp_l] = top_name
            print(f'    ↳ auto-accepted')
        elif top_score >= 0.70:
            # Suggest but don't auto-accept
            new_aliases.setdefault(pp_l, top_name)

    return new_aliases, suggestions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Resolve PrizePicks → VLR player name aliases')
    parser.add_argument('--min-appearances', type=int, default=15,
                        help='Flag players with fewer than N map appearances (default: 15)')
    parser.add_argument('--league-id', type=int, default=159,
                        help='PrizePicks league ID (default: 159)')
    parser.add_argument('--all', action='store_true',
                        help='Show all PP names, not just low-appearance ones')
    args = parser.parse_args()

    print()
    print('══════════════════════════════════════════════════')
    print(' PLAYER NAME RESOLVER')
    print('══════════════════════════════════════════════════')

    print('\n[1] Loading VLR player names from DB...')
    vlr_names   = load_vlr_names()
    appearances = load_appearances_from_db()
    print(f'    {len(vlr_names):,} unique VLR names  |  {len(appearances):,} with match data')

    print('\n[2] Fetching live PrizePicks lines...')
    client    = PrizePicksClient(league_id=args.league_id)
    lines     = client.fetch_kill_lines()
    pp_names  = list(lines.keys())
    if not pp_names:
        print('    No active lines found.')
        return
    print(f'    {len(pp_names)} active players: {", ".join(sorted(pp_names))}')

    print('\n[3] Loading existing aliases...')
    existing = load_aliases()
    print(f'    {len(existing)} existing alias entries')

    print(f'\n[4] Resolving names (min_appearances={args.min_appearances})...')
    new_aliases, suggestions = resolve(
        pp_names, args.min_appearances, vlr_names, appearances, existing
    )

    print('\n[5] Writing name_aliases.json...')
    save_aliases(new_aliases, suggestions)
    print(f'    Saved → {ALIASES_FILE}')

    # Summary
    flagged = [p for p in pp_names if appearances.get(p.lower(), 0) < args.min_appearances
               and p.lower() not in existing]
    print()
    print(f'  {len(pp_names)} PP players checked')
    print(f'  {len(flagged)} flagged as low/no data')
    print(f'  {len(new_aliases)} alias entries total (including existing)')
    print()
    print('  Review name_aliases.json — correct any wrong suggestions,')
    print('  then rerun bet_slate.py to pick up the changes.')
    print()


if __name__ == '__main__':
    main()
