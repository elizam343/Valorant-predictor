import requests
import db_utils
import time
from flask import Flask, jsonify
from bs4 import BeautifulSoup

API_BASE_URL = "https://vlrggapi.vercel.app"
VLR_BASE_URL = "https://www.vlr.gg"

# All available regions
REGIONS = ["na", "eu", "ap", "kr", "br", "latam", "oce", "mn", "gc", "cn"]

# Flask app for API endpoints
app = Flask(__name__)

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

# --- NEW: Scrape full match details from VLR.gg ---
def scrape_match_details(match_id):
    url = f"{VLR_BASE_URL}/{match_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    match_data = {"match_id": match_id}
    
    # Teams - try multiple selectors
    teams = []
    # Try primary selector
    team_elements = soup.select(".match-header-vs-team-name")
    if team_elements:
        teams = [t.text.strip() for t in team_elements]
    else:
        # Try alternative selectors
        team_elements = soup.select(".match-header-vs-team")
        if team_elements:
            teams = [t.text.strip() for t in team_elements]
        else:
            # Try finding teams in the match header
            team_elements = soup.select(".match-header .team-name")
            if team_elements:
                teams = [t.text.strip() for t in team_elements]
    
    match_data["teams"] = teams
    
    # Date, tournament
    match_data["date"] = soup.select_one(".moment-tz-convert").get("data-utc-ts", "") if soup.select_one(".moment-tz-convert") else ""
    match_data["tournament"] = soup.select_one(".match-header-event-series").text.strip() if soup.select_one(".match-header-event-series") else ""
    
    # Maps
    maps = [m.text.strip().split("\n")[0].strip() for m in soup.select(".vm-stats-gamesnav .map")]
    match_data["maps"] = maps
    
    # Player stats per map
    match_data["map_stats"] = []
    map_idx = 0
    all_teams_found = set()  # Track all teams found in player data
    
    for map_tab in soup.select(".vm-stats-game"):  # Each map
        map_name = map_tab.select_one(".map").text.strip() if map_tab.select_one(".map") else ""
        map_name = map_name.split("\n")[0].strip()  # Clean up map name
        players = []
        team_names = [t.text.strip() for t in map_tab.select(".team-name")]
        team_idx = 0
        for team_table in map_tab.select(".wf-table-inset"):  # Each team
            team_players = []
            team_name = team_names[team_idx] if team_idx < len(team_names) else None
            for row in team_table.select("tbody tr"):
                cols = [td.text.strip() for td in row.select("td")]
                if len(cols) >= 10:
                    # Split kills by newlines if present
                    kills_split = [k.strip() for k in cols[3].split("\n") if k.strip()]
                    # Use the first value if multiple, or the only value
                    kills = kills_split[map_idx] if map_idx < len(kills_split) else (kills_split[0] if kills_split else cols[3])
                    player = {
                        "name": cols[0].split("\n")[0].strip(),
                        "agent": cols[1],
                        "acs": cols[2],
                        "kills": kills,
                        "deaths": cols[4],
                        "assists": cols[5],
                        "kdr": cols[6],
                        "adr": cols[7],
                        "hs%": cols[8],
                        "fk": cols[9],
                        "team": team_name
                    }
                    team_players.append(player)
                    if team_name:
                        all_teams_found.add(team_name)
            players.append(team_players)
            team_idx += 1
        
        # --- NEW: Extract round-by-round results and half-time scores ---
        round_results = []
        halftime_score = None
        round_timeline = map_tab.select_one(".scoreboard-rounds")
        if round_timeline:
            round_icons = round_timeline.select(".scoreboard-round")
            for icon in round_icons:
                winner = None
                if "left" in icon.get("class", []):
                    winner = "team1"
                elif "right" in icon.get("class", []):
                    winner = "team2"
                round_results.append(winner)
            if len(round_results) >= 12:
                halftime_score = {
                    "team1": round_results[:12].count("team1"),
                    "team2": round_results[:12].count("team2")
                }
        total_score = {
            "team1": round_results.count("team1"),
            "team2": round_results.count("team2")
        } if round_results else None
        # --- NEW: Add flat player list for this map ---
        flat_players = []
        for team in players:
            for p in team:
                flat_players.append({
                    "name": p["name"],
                    "kills": p["kills"],
                    "team": p["team"]
                })
        match_data["map_stats"].append({
            "map": map_name,
            "players": players,
            "flat_players": flat_players,
            "round_results": round_results,
            "halftime_score": halftime_score,
            "total_score": total_score
        })
        map_idx += 1
    
    # If we didn't find teams in header but found them in player data, use those
    if not teams and all_teams_found:
        teams = list(all_teams_found)
        match_data["teams"] = teams
    
    return match_data

@app.route("/match/<int:match_id>")
def api_match(match_id):
    match_details = scrape_match_details(match_id)
    return jsonify(match_details)

if __name__ == "__main__":
    # Only create tables if needed, but do not write any player or match data
    # db_utils.create_tables()  # Optional: comment out if tables are already created
    app.run(debug=True, host="0.0.0.0", port=5003)
