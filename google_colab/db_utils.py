import sqlite3

import os

# Get the absolute path to the database file
DB_PATH = os.path.join(os.path.dirname(__file__), "vlr_players.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            team TEXT NOT NULL,
            rating TEXT,
            average_combat_score TEXT,
            kill_deaths TEXT,
            kill_assists_survived_traded TEXT,
            average_damage_per_round TEXT,
            kills_per_round TEXT,
            assists_per_round TEXT,
            first_kills_per_round TEXT,
            first_deaths_per_round TEXT,
            headshot_percentage TEXT,
            clutch_success_percentage TEXT,
            UNIQUE(name, team)
        )
    """)
    conn.commit()
    conn.close()

def has_verified_team(team):
    return team and len(team) > 2 and not team.isdigit() and not team.startswith('(+')

def is_coach_or_analyst(player_name, team):
    return (
        'coach' in player_name.lower() or 'analyst' in player_name.lower() or
        'coach' in team.lower() or 'analyst' in team.lower()
    )

def has_match_history(player):
    # Use kills_per_round, average_combat_score, kill_deaths as proxy for match history
    try:
        kpr = float(player.get("kills_per_round", 0) or 0)
        acs = float(player.get("average_combat_score", 0) or 0)
        kd = float(player.get("kill_deaths", 0) or 0)
        return (kpr > 0 or acs > 0 or kd > 0)
    except Exception:
        return False

def upsert_player(player):
    name = player.get("player")
    team = player.get("org")
    if not has_verified_team(team):
        print(f"[SKIP] {name} ({team}): Unverified team")
        return
    if is_coach_or_analyst(name, team):
        print(f"[SKIP] {name} ({team}): Coach/Analyst")
        return
    if not has_match_history(player):
        print(f"[SKIP] {name} ({team}): No match history")
        return
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO players (
            name, team, rating, average_combat_score, kill_deaths, kill_assists_survived_traded,
            average_damage_per_round, kills_per_round, assists_per_round, first_kills_per_round,
            first_deaths_per_round, headshot_percentage, clutch_success_percentage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name, team) DO UPDATE SET
            rating=excluded.rating,
            average_combat_score=excluded.average_combat_score,
            kill_deaths=excluded.kill_deaths,
            kill_assists_survived_traded=excluded.kill_assists_survived_traded,
            average_damage_per_round=excluded.average_damage_per_round,
            kills_per_round=excluded.kills_per_round,
            assists_per_round=excluded.assists_per_round,
            first_kills_per_round=excluded.first_kills_per_round,
            first_deaths_per_round=excluded.first_deaths_per_round,
            headshot_percentage=excluded.headshot_percentage,
            clutch_success_percentage=excluded.clutch_success_percentage
    """, (
        name,
        team,
        player.get("rating"),
        player.get("average_combat_score"),
        player.get("kill_deaths"),
        player.get("kill_assists_survived_traded"),
        player.get("average_damage_per_round"),
        player.get("kills_per_round"),
        player.get("assists_per_round"),
        player.get("first_kills_per_round"),
        player.get("first_deaths_per_round"),
        player.get("headshot_percentage"),
        player.get("clutch_success_percentage")
    ))
    conn.commit()
    conn.close()

def get_players():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM players ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return rows

def get_teams():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT team FROM players ORDER BY team")
    teams = [row[0] for row in c.fetchall()]
    conn.close()
    return teams

def get_players_by_team(team):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM players WHERE team = ? ORDER BY name", (team,))
    players = c.fetchall()
    conn.close()
    return players 