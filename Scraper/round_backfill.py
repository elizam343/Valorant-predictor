"""Round-count Stage 1b: bounded, resumable backfill of TRUE map scores (#9).

Enumerates recent completed matches from VLR's results listing (only real matches
— no wasted ID scanning), then scrapes each with the fixed parser and saves the
scored JSON to scraped_matches_scored/. Resumable (skips already-scored matches),
rate-limited, polite UA. Run as a background job.

Usage:  python round_backfill.py [PAGES]      # default 50 pages (~2500 matches)
"""
import os, re, sys, json, time
import requests
from bs4 import BeautifulSoup
import scraper_api

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'scraped_matches_scored')
UA = {"User-Agent": "Mozilla/5.0 (compatible; valorant-research/1.0)"}
SLEEP = 0.6


def enumerate_ids(pages):
    """Collect unique match IDs from results pages 1..pages (newest first)."""
    ids = []
    for p in range(1, pages + 1):
        try:
            html = requests.get(f"https://www.vlr.gg/matches/results?page={p}",
                                headers=UA, timeout=15).text
        except Exception:
            time.sleep(SLEEP); continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/(\d+)/", a["href"])     # /684615/... = a match page
            if m:
                ids.append(int(m.group(1)))
        if p % 10 == 0:
            print(f"  enumerated {p}/{pages} pages, {len(set(ids))} unique ids", flush=True)
        time.sleep(SLEEP)
    return sorted(set(ids), reverse=True)


def backfill(pages):
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = enumerate_ids(pages)
    print(f"\n{len(ids)} unique match ids to process\n", flush=True)

    scraped = skipped = maps_scored = no_rounds = 0
    for i, mid in enumerate(ids):
        out = os.path.join(OUT_DIR, f"match_{mid}.json")
        if os.path.exists(out):
            skipped += 1
            continue
        try:
            d = scraper_api.scrape_match_details(mid)
            time.sleep(SLEEP)
        except Exception:
            time.sleep(SLEEP); continue
        nscored = sum(1 for m in (d.get("map_stats") or []) if m.get("total_score"))
        if nscored:
            json.dump(d, open(out, "w"))
            scraped += 1
            maps_scored += nscored
        else:
            no_rounds += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ids)} | new {scraped}, skip {skipped}, "
                  f"no-rounds {no_rounds} | {maps_scored} maps w/ true rounds", flush=True)

    total = len([f for f in os.listdir(OUT_DIR) if f.endswith('.json')])
    print(f"\n{'='*56}")
    print(f"  BACKFILL DONE")
    print(f"{'='*56}")
    print(f"  newly scraped : {scraped} matches  ({maps_scored} maps with true rounds)")
    print(f"  already had   : {skipped}")
    print(f"  no round data : {no_rounds} (incomplete/forfeit/old)")
    print(f"  total scored JSONs in scraped_matches_scored/: {total}")


if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Stage 1b backfill: {pages} results pages (~{pages*50} matches), sleep {SLEEP}s")
    backfill(pages)
