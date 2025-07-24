import requests
import time
from database_schema import ValorantDatabase

# =========================================================================
# SCRAPER PROGRESS TRACKING
# =========================================================================
# Last run completed: July 19, 2025
# Last scraped match ID: 169058
# Total matches scraped: 34,596
# 
# To resume scraping:
# 1. Update START_ID below to 169059 (last scraped + 1)
# 2. Run this script again
# =========================================================================

API_URL = "http://localhost:5003/match/{}"
START_ID = 169071  # Start from the next match after the last scraped
END_ID = 520000  # Set high, adjust as needed
SLEEP_BETWEEN = 0.1  # Reduced sleep time
SLEEP_BETWEEN_VALID = 0.3  # Longer sleep for valid matches

db = ValorantDatabase("Scraper/valorant_matches.db")

def quick_match_check(match_id):
    """Quick check if a match ID might be valid before full scraping"""
    url = f"https://www.vlr.gg/{match_id}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; VLR-Scraper/1.0)'}
        resp = requests.get(url, timeout=5, headers=headers)
        if resp.status_code != 200:
            return False
        
        content = resp.text.lower()
        
        # Only skip pages that are clearly NOT match pages
        if any(indicator in content for indicator in [
            'general discussion', 'forum thread', 'forum topic', 'forum post',
            'page not found', '404 error', 'invalid page', 'not found'
        ]):
            return False
        
        # For everything else, assume it might be a match and do full check
        return True
        
    except Exception as e:
        # If there's any error, assume it might be valid and do full check
        return True

def is_valid_match(data):
    # Consider a match valid if it has at least two teams and at least one map
    if not data or "teams" not in data or len(data["teams"]) != 2:
        return False
    if "map_stats" not in data or not data["map_stats"]:
        return False
    # If the page is a general discussion or not a match, teams will be empty or not 2
    return True

def scrape_and_save(match_id):
    # Quick check first
    if not quick_match_check(match_id):
        print(f"[SKIP] {match_id}: No match indicators found")
        return False
    
    # Full scrape only for potential matches
    url = API_URL.format(match_id)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"[SKIP] {match_id}: HTTP {resp.status_code}")
            return False
        data = resp.json()
        if not is_valid_match(data):
            print(f"[SKIP] {match_id}: Not a match or invalid data")
            return False
        # Insert into database instead of saving JSON
        try:
            db.insert_match(data)
            print(f"[OK] {match_id}: Inserted into database")
            return True
        except Exception as db_exc:
            print(f"[DB ERR] {match_id}: {db_exc}")
            return False
    except Exception as e:
        print(f"[ERR] {match_id}: {e}")
        return False

def main():
    for match_id in range(START_ID, END_ID):
        result = scrape_and_save(match_id)
        # Use different sleep times based on result
        if result:
            time.sleep(SLEEP_BETWEEN_VALID)  # Longer sleep for valid matches
        else:
            time.sleep(SLEEP_BETWEEN)  # Quick sleep for invalid matches

if __name__ == "__main__":
    main() 