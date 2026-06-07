import sqlite3
import os

def get_connection():
    db_path = os.path.join(os.path.dirname(__file__), "vlr_players.db")
    return sqlite3.connect(db_path)

def get_all_players():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM players")
    rows = c.fetchall()
    conn.close()
    return rows

def get_player_match_history(player_name, team):
    # Placeholder: In a real system, you would join with a match history table
    # For now, we assume that if a player has nonzero stats, they have match history
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT kills_per_round, average_combat_score, kill_deaths
        FROM players WHERE name = ? AND team = ?
    """, (player_name, team))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    # If all stats are zero or None, treat as no match history
    return any(row) and all((v is not None and float(v) > 0) for v in row)

def is_coach_or_analyst(player_name, team):
    # Placeholder: In a real system, you would check a role field or external data
    # For now, we use a heuristic: if 'coach' or 'analyst' in team or player name
    return (
        'coach' in player_name.lower() or 'analyst' in player_name.lower() or
        'coach' in team.lower() or 'analyst' in team.lower()
    )

def has_verified_team(team):
    # Placeholder: In a real system, you would check against a list of verified teams
    # For now, we treat teams with length > 2 and not all digits as verified
    return team and len(team) > 2 and not team.isdigit() and not team.startswith('(+')

def clean_players_db():
    players = get_all_players()
    to_remove = []
    for row in players:
        player_id, name, team = row[0], row[1], row[2]
        if not has_verified_team(team):
            print(f"[REMOVE] {name} ({team}): Unverified team")
            to_remove.append(player_id)
            continue
        if is_coach_or_analyst(name, team):
            print(f"[REMOVE] {name} ({team}): Coach/Analyst")
            to_remove.append(player_id)
            continue
        if not get_player_match_history(name, team):
            print(f"[REMOVE] {name} ({team}): No match history")
            to_remove.append(player_id)
            continue
    # Remove flagged players
    if to_remove:
        conn = get_connection()
        c = conn.cursor()
        c.executemany("DELETE FROM players WHERE id = ?", [(pid,) for pid in to_remove])
        conn.commit()
        conn.close()
        print(f"Removed {len(to_remove)} players from the database.")
    else:
        print("No players removed. Database is clean.")

def main():
    print("Scanning and cleaning player database...")
    clean_players_db()
    print("Done.")

if __name__ == "__main__":
    main() 