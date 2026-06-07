#!/usr/bin/env python3
"""
One-shot script to backfill full player stats into flat_players for all existing
scraped match JSON files. The original scraper only stored name/kills/team in
flat_players; the full stats are in the nested players array but with multi-map
values concatenated by newlines. This script picks the correct map-index value
and writes all stats back into flat_players.

Usage:
    python patch_existing_matches.py [--dry-run] [--limit N]
"""

import json
import glob
import os
import argparse
from tqdm import tqdm


def split_stat(raw, map_idx):
    """Return the value for map_idx from a newline-separated multi-map cell."""
    parts = [p.strip() for p in str(raw).split("\n") if p.strip()]
    if map_idx < len(parts):
        return parts[map_idx]
    return parts[0] if parts else "0"


def safe_int(value, default=0):
    try:
        return int(str(value).replace("/", "").strip() or default)
    except (ValueError, TypeError):
        return default


def patch_match(data, filepath):
    """Patch a single match dict in-place. Returns True if any change was made."""
    changed = False
    for map_idx, map_stat in enumerate(data.get("map_stats", [])):
        nested_players = map_stat.get("players", [])
        if not nested_players:
            continue

        # Build a lookup from player name to full stats from the nested array
        name_to_stats = {}
        for team_players in nested_players:
            for p in team_players:
                name = p.get("name", "").strip()
                if name:
                    name_to_stats[name] = p

        new_flat = []
        for fp in map_stat.get("flat_players", []):
            name = fp.get("name", "").strip()
            src = name_to_stats.get(name, fp)

            # Determine column layout: old scraper used wrong column order.
            # New scraper order: rating, acs, kills, deaths, assists, kd_diff,
            #                    kast, adr, hs%, fk, fd
            # Old scraper stored: acs(=rating), kills, deaths, assists, kdr, adr, hs%, fk
            # Detect which format we have by checking for the 'rating' key.
            if "rating" in src:
                # New format — already correct
                rating  = split_stat(src.get("rating", "0"), map_idx)
                acs     = split_stat(src.get("acs", "0"), map_idx)
                kills   = split_stat(src.get("kills", "0"), map_idx)
                deaths  = split_stat(src.get("deaths", "0"), map_idx)
                assists = split_stat(src.get("assists", "0"), map_idx)
                kd_diff = split_stat(src.get("kd_diff", "0"), map_idx)
                kast    = split_stat(src.get("kast", "0"), map_idx)
                adr     = split_stat(src.get("adr", "0"), map_idx)
                hs_pct  = split_stat(src.get("hs%", "0%"), map_idx)
                fk      = split_stat(src.get("fk", "0"), map_idx)
                fd      = split_stat(src.get("fd", "0"), map_idx)
                agent   = src.get("agent", "")
            else:
                # Old format: the scraper's column labels were shifted by 2.
                # Actual VLR.gg columns: Rating, ACS, Kills, Deaths, Assists, +/-, KAST, ADR, HS%, FK, FD
                # Old scraper stored:    acs,    kills, deaths, assists, kdr,   adr,  hs%,  fk
                # So old "acs"     → rating  (1.xx VLR rating)
                #    old "kills"   → acs     (ACS like 250)
                #    old "deaths"  → kills   (per-map kills like 23)
                #    old "assists" → deaths  (per-map deaths)
                #    old "kdr"     → assists
                #    old "adr"     → kd_diff (+/- differential)
                #    old "hs%"     → kast    (KAST %)
                #    old "fk"      → adr     (ADR like 187)
                rating  = split_stat(src.get("acs", "0"), map_idx)
                acs     = split_stat(src.get("kills", "0"), map_idx)
                kills   = split_stat(src.get("deaths", "0"), map_idx)
                deaths  = split_stat(src.get("assists", "0"), map_idx)
                assists = split_stat(src.get("kdr", "0"), map_idx)
                kd_diff = split_stat(src.get("adr", "0"), map_idx)
                kast    = split_stat(src.get("hs%", "0%"), map_idx)
                adr     = split_stat(src.get("fk", "0"), map_idx)
                hs_pct  = "0%"
                fk      = "0"
                fd      = "0"
                agent   = src.get("agent", "")

            patched = {
                "name":    name,
                "team":    fp.get("team", src.get("team", "")),
                "agent":   agent,
                "rating":  rating,
                "acs":     acs,
                "kills":   kills,
                "deaths":  deaths,
                "assists": assists,
                "kd_diff": kd_diff,
                "kast":    kast,
                "adr":     adr,
                "hs%":     hs_pct,
                "fk":      fk,
                "fd":      fd,
            }

            if patched != fp:
                changed = True
            new_flat.append(patched)

        map_stat["flat_players"] = new_flat

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and patch in memory but do not write files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files (for testing)")
    args = parser.parse_args()

    matches_dir = os.path.join(os.path.dirname(__file__), "scraped_matches")
    files = sorted(glob.glob(os.path.join(matches_dir, "match_*.json")))

    if args.limit:
        files = files[: args.limit]

    patched_count = 0
    error_count = 0

    for filepath in tqdm(files, desc="Patching JSONs"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            changed = patch_match(data, filepath)

            if changed and not args.dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                patched_count += 1
            elif changed:
                patched_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {filepath}: {e}")

    print(f"\nDone. Patched: {patched_count}, Errors: {error_count}, Total: {len(files)}")
    if args.dry_run:
        print("(dry-run — no files were written)")


if __name__ == "__main__":
    main()
