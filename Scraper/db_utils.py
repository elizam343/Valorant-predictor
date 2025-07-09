import sqlite3

DB_PATH = "vlr_players.db"

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

def upsert_player(player):
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
        player.get("player"),
        player.get("org"),
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