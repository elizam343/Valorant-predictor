import requests
import db_utils
import time

API_BASE_URL = "https://vlrggapi.vercel.app"

# All available regions
REGIONS = ["na", "eu", "ap", "kr", "br", "latam", "oce", "mn", "gc", "cn"]

# Fetch player stats for a given region and timespan
def fetch_players(region="na", timespan="all"):
    url = f"{API_BASE_URL}/stats?region={region}&timespan={timespan}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        players = data.get("data", {}).get("segments", [])
        return players
    except Exception as e:
        print(f"Error fetching data from region {region}: {e}")
        return []

def fetch_all_regions(timespan="all"):
    """Fetch player data from all regions"""
    all_players = []
    
    for region in REGIONS:
        print(f"Fetching data from region: {region}")
        try:
            players = fetch_players(region, timespan)
            if players:
                print(f"  Found {len(players)} players in {region}")
                all_players.extend(players)
            else:
                print(f"  No players found in {region}")
            
            # Add a small delay to be respectful to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"  Failed to fetch from {region}: {e}")
            continue
    
    return all_players

if __name__ == "__main__":
    db_utils.create_tables()
    
    print("Fetching player data from all regions...")
    all_players = fetch_all_regions()
    
    print(f"\nTotal players fetched from all regions: {len(all_players)}")
    
    # Update database with all players
    updated_count = 0
    for player in all_players:
        try:
            db_utils.upsert_player(player)
            updated_count += 1
        except Exception as e:
            print(f"Error updating player {player.get('player', 'Unknown')}: {e}")
    
    print(f"Database updated with {updated_count} player records from all regions.")
