#!/usr/bin/env python3
"""
results_scraper.py — Fast parallel VLR.gg match scraper.

Strategy
--------
Instead of scanning every integer match ID (slow, ~50% miss rate), paginate
vlr.gg/matches/results to get only real completed-match URLs, then scrape
each one in a small thread pool.

Compared to bulk_scrape_matches.py:
  • Zero wasted HTTP requests on non-match IDs
  • Parallel workers — default 5, tunable with --workers
  • Checkpoint file so you can stop and resume without re-scraping
  • Newest matches first, so recent data lands immediately

Output
------
  scraped_matches/match_<ID>.json   (same format as existing files)
  Scraper/valorant_matches.db       (same schema as before)

Usage
-----
    cd "Data annotation raven project/valorant kill line predicter"
    python Scraper/results_scraper.py                   # run everything
    python Scraper/results_scraper.py --workers 8       # more parallelism
    python Scraper/results_scraper.py --max-pages 10    # quick test
    python Scraper/results_scraper.py --start-page 50   # skip first 49 pages
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(__file__))
from database_schema import ValorantDatabase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).parent
_REPO_ROOT  = _HERE.parent
JSON_DIR    = _REPO_ROOT / 'scraped_matches'
DB_PATH     = _HERE / 'valorant_matches.db'
CHECKPOINT  = _HERE / '.results_scraper_checkpoint.json'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLR_BASE    = 'https://www.vlr.gg'
RESULTS_URL = f'{VLR_BASE}/matches/results'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Referer': 'https://www.vlr.gg/',
}

DEFAULT_WORKERS = 5
PAGE_DELAY      = (1.5, 2.5)   # seconds between index-page fetches (be polite)
MATCH_DELAY     = (0.5, 1.0)   # seconds per match worker before hitting VLR
TIMEOUT         = 15

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('results_scraper')


# ---------------------------------------------------------------------------
# Checkpoint (thread-safe)
# ---------------------------------------------------------------------------

class Checkpoint:
    """
    Persists scraped match IDs to disk so runs can be interrupted and resumed.
    Uses a plain JSON file; the set of IDs is kept in memory for O(1) lookup.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        raw = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
            except Exception:
                pass
        self._done: set[int] = set(raw.get('scraped', []))
        self._last_page: int = raw.get('last_page', 0)

    def _flush(self):
        self.path.write_text(json.dumps(
            {'scraped': sorted(self._done), 'last_page': self._last_page},
            indent=2,
        ))

    def already_done(self, match_id: int) -> bool:
        with self._lock:
            return match_id in self._done

    def mark_done(self, match_id: int):
        with self._lock:
            self._done.add(match_id)
            self._flush()

    def set_last_page(self, page: int):
        with self._lock:
            self._last_page = page
            self._flush()

    @property
    def last_page(self) -> int:
        return self._last_page

    @property
    def total(self) -> int:
        return len(self._done)


# ---------------------------------------------------------------------------
# Step 1: collect match IDs from the results index
# ---------------------------------------------------------------------------

def fetch_match_ids_from_page(page: int, session: requests.Session) -> list[int]:
    """
    Scrape one page of vlr.gg/matches/results.
    Returns a deduplicated list of integer match IDs in page order.
    Returns [] on error or when the page is empty (signals end of results).
    """
    url = RESULTS_URL if page == 1 else f'{RESULTS_URL}?page={page}'
    try:
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            log.warning(f'Results page {page}: HTTP {resp.status_code}')
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Match cards are <a class="match-item" href="/MATCHID/team-a-vs-team-b/...">
        ids: list[int] = []
        seen: set[int] = set()

        for a in soup.select('a.match-item'):
            href = a.get('href', '').strip('/')
            first = href.split('/')[0]
            if first.isdigit():
                mid = int(first)
                if mid > 0 and mid not in seen:
                    seen.add(mid)
                    ids.append(mid)

        # Fallback: any <a href="/DIGITS/..."> if the selector changed
        if not ids:
            for a in soup.select('a[href]'):
                href = a.get('href', '').strip('/')
                parts = href.split('/')
                if len(parts) >= 2 and parts[0].isdigit():
                    mid = int(parts[0])
                    if mid > 10000 and mid not in seen:  # filter out nav links (< 10K)
                        seen.add(mid)
                        ids.append(mid)

        return ids

    except Exception as e:
        log.error(f'Results page {page}: {e}')
        return []


# ---------------------------------------------------------------------------
# Step 2: scrape individual match pages
# ---------------------------------------------------------------------------

def _split_stat(raw: str, idx: int) -> str:
    """Pick the idx-th non-empty line from a multi-line cell value."""
    parts = [p.strip() for p in str(raw).split('\n') if p.strip()]
    if idx < len(parts):
        return parts[idx]
    return parts[0] if parts else '0'


def scrape_match_page(match_id: int) -> dict | None:
    """
    Fetch and parse a single VLR.gg match page.
    Returns the match dict (same structure as scraper_api.scrape_match_details)
    or None if the page isn't a valid completed match.
    """
    url = f'{VLR_BASE}/{match_id}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        log.debug(f'[{match_id}] network error: {e}')
        return None

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')
    match_data: dict = {'match_id': match_id}

    # ── Teams ──────────────────────────────────────────────────────────────
    teams = [t.text.strip() for t in soup.select('.match-header-vs-team-name')]
    if not teams:
        teams = [t.text.strip() for t in soup.select('.match-header-vs-team')]
    match_data['teams'] = teams

    # ── Date / tournament ──────────────────────────────────────────────────
    dt_el = soup.select_one('.moment-tz-convert')
    match_data['date'] = dt_el.get('data-utc-ts', '') if dt_el else ''
    tour_el = soup.select_one('.match-header-event-series')
    match_data['tournament'] = tour_el.text.strip() if tour_el else ''

    # ── Per-map stats ──────────────────────────────────────────────────────
    match_data['map_stats'] = []
    all_teams_found: set[str] = set()

    for map_idx, map_tab in enumerate(soup.select('.vm-stats-game')):
        map_el = map_tab.select_one('.map')
        map_name = map_el.text.strip().split('\n')[0].strip() if map_el else ''

        team_names = [t.text.strip() for t in map_tab.select('.team-name')]
        players_by_team: list[list[dict]] = []
        flat_players: list[dict] = []

        for team_idx, team_table in enumerate(map_tab.select('.wf-table-inset')):
            team_name = team_names[team_idx] if team_idx < len(team_names) else None
            team_players: list[dict] = []

            for row in team_table.select('tbody tr'):
                tds = row.select('td')
                if len(tds) < 10:
                    continue
                cols = [td.text.strip() for td in tds]

                # Agent is an image — extract from img[alt] or span[title]
                agent = ''
                agent_td = tds[1]
                img = agent_td.find('img')
                if img:
                    agent = img.get('alt') or img.get('title') or ''
                if not agent:
                    span = agent_td.find('span')
                    if span:
                        agent = span.get('title') or ''

                player = {
                    'name':    cols[0].split('\n')[0].strip(),
                    'agent':   agent.strip(),
                    'rating':  _split_stat(cols[2],  map_idx),
                    'acs':     _split_stat(cols[3],  map_idx),
                    'kills':   _split_stat(cols[4],  map_idx),
                    'deaths':  _split_stat(cols[5],  map_idx),
                    'assists': _split_stat(cols[6],  map_idx),
                    'kd_diff': _split_stat(cols[7],  map_idx),
                    'kast':    _split_stat(cols[8],  map_idx),
                    'adr':     _split_stat(cols[9],  map_idx),
                    'hs%':     _split_stat(cols[10], map_idx) if len(cols) > 10 else '0%',
                    'fk':      _split_stat(cols[11], map_idx) if len(cols) > 11 else '0',
                    'fd':      _split_stat(cols[12], map_idx) if len(cols) > 12 else '0',
                    'team':    team_name,
                }
                team_players.append(player)
                flat_players.append({k: v for k, v in player.items()})
                if team_name:
                    all_teams_found.add(team_name)

            players_by_team.append(team_players)

        # Round-by-round timeline
        round_results: list[str | None] = []
        halftime_score = None
        timeline = map_tab.select_one('.scoreboard-rounds')
        if timeline:
            for icon in timeline.select('.scoreboard-round'):
                cls = icon.get('class', [])
                if 'left' in cls:
                    round_results.append('team1')
                elif 'right' in cls:
                    round_results.append('team2')
                else:
                    round_results.append(None)
            if len(round_results) >= 12:
                halftime_score = {
                    'team1': round_results[:12].count('team1'),
                    'team2': round_results[:12].count('team2'),
                }

        total_score = (
            {'team1': round_results.count('team1'), 'team2': round_results.count('team2')}
            if round_results else None
        )

        match_data['map_stats'].append({
            'map':            map_name,
            'players':        players_by_team,
            'flat_players':   flat_players,
            'round_results':  round_results,
            'halftime_score': halftime_score,
            'total_score':    total_score,
        })

    # Use player-derived teams if header teams were missing
    if not match_data['teams'] and all_teams_found:
        match_data['teams'] = list(all_teams_found)

    # Validity check: need 2 teams and at least 1 completed map with players
    if len(match_data['teams']) < 2:
        return None
    valid_maps = [m for m in match_data['map_stats'] if m['flat_players']]
    if not valid_maps:
        return None

    return match_data


# ---------------------------------------------------------------------------
# Step 3: save worker
# ---------------------------------------------------------------------------

def _json_path(match_id: int) -> Path:
    return JSON_DIR / f'match_{match_id}.json'


def scrape_and_save(match_id: int, db: ValorantDatabase) -> bool:
    """
    Thread-worker: scrape one match and persist it.
    Returns True on success (including if already on disk).
    """
    # Already saved from a previous run?
    out = _json_path(match_id)
    if out.exists():
        return True

    time.sleep(random.uniform(*MATCH_DELAY))   # polite random stagger

    data = scrape_match_page(match_id)
    if data is None:
        return False

    # JSON file
    try:
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        log.warning(f'[{match_id}] JSON write failed: {e}')
        return False

    # Database (best-effort — JSON file is the source of truth)
    try:
        db.insert_match(data)
    except Exception:
        pass

    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ParallelScraper:
    def __init__(
        self,
        workers:    int = DEFAULT_WORKERS,
        start_page: int = 1,
        max_pages:  int | None = None,
    ):
        self.workers    = workers
        self.start_page = start_page
        self.max_pages  = max_pages
        self.ckpt       = Checkpoint(CHECKPOINT)
        self.db         = ValorantDatabase(str(DB_PATH))
        self._session   = requests.Session()
        self._lock      = threading.Lock()
        self._stats     = {'scraped': 0, 'skipped': 0, 'failed': 0}

    def _update(self, key: str, n: int = 1):
        with self._lock:
            self._stats[key] += n

    def _banner(self, page: int):
        s = self._stats
        log.info(
            f'Page {page:4d} complete | '
            f'+scraped={s["scraped"]:,} | '
            f'skipped={s["skipped"]:,} | '
            f'failed={s["failed"]:,} | '
            f'checkpoint total={self.ckpt.total:,}'
        )

    def run(self):
        JSON_DIR.mkdir(parents=True, exist_ok=True)

        log.info('─' * 60)
        log.info('VLR.gg Results Scraper (parallel)')
        log.info(f'  Workers    : {self.workers}')
        log.info(f'  Start page : {self.start_page}')
        log.info(f'  Max pages  : {self.max_pages or "unlimited"}')
        log.info(f'  JSON dir   : {JSON_DIR}')
        log.info(f'  Database   : {DB_PATH}')
        log.info(f'  Checkpoint : {self.ckpt.total:,} matches already done')
        log.info('─' * 60)

        page = self.start_page
        pages_done = 0

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while True:
                if self.max_pages and pages_done >= self.max_pages:
                    log.info(f'Reached --max-pages {self.max_pages}. Stopping.')
                    break

                match_ids = fetch_match_ids_from_page(page, self._session)

                if not match_ids:
                    log.info(f'Page {page}: no match IDs returned — end of results.')
                    break

                log.info(f'Page {page}: found {len(match_ids)} match IDs')

                # Filter already done (checkpoint + existing JSON files)
                to_scrape = [
                    mid for mid in match_ids
                    if not self.ckpt.already_done(mid) and not _json_path(mid).exists()
                ]
                self._update('skipped', len(match_ids) - len(to_scrape))

                # Submit batch
                futures = {pool.submit(scrape_and_save, mid, self.db): mid
                           for mid in to_scrape}

                for fut in as_completed(futures):
                    mid = futures[fut]
                    try:
                        ok = fut.result()
                    except Exception:
                        ok = False
                    if ok:
                        self.ckpt.mark_done(mid)
                        self._update('scraped')
                    else:
                        self._update('failed')

                self.ckpt.set_last_page(page)
                self._banner(page)

                pages_done += 1
                page += 1

                # Polite delay between index-page fetches
                time.sleep(random.uniform(*PAGE_DELAY))

        log.info('═' * 60)
        s = self._stats
        log.info(
            f'Run complete — scraped: {s["scraped"]:,}  '
            f'skipped: {s["skipped"]:,}  failed: {s["failed"]:,}'
        )
        log.info(f'Checkpoint total: {self.ckpt.total:,} matches')
        log.info('═' * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fast parallel VLR.gg results scraper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--workers', type=int, default=DEFAULT_WORKERS,
        help='Parallel worker threads',
    )
    parser.add_argument(
        '--start-page', type=int, default=None,
        help='Results page to start from (default: resume from checkpoint, or 1)',
    )
    parser.add_argument(
        '--max-pages', type=int, default=None,
        help='Stop after this many pages (useful for testing)',
    )
    args = parser.parse_args()

    ckpt = Checkpoint(CHECKPOINT)
    start_page = args.start_page or max(1, ckpt.last_page)

    scraper = ParallelScraper(
        workers=args.workers,
        start_page=start_page,
        max_pages=args.max_pages,
    )
    scraper.run()


if __name__ == '__main__':
    main()
