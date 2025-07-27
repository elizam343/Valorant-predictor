import os
import sqlite3

def check_database_schema(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players';")
        result = cursor.fetchone()
        if not result:
            print("ERROR: 'players' table not found in the database.")
            conn.close()
            return False
        cursor.execute("PRAGMA table_info(players);")
        columns = [row[1] for row in cursor.fetchall()]
        required_columns = [
            'name', 'team', 'rating', 'average_combat_score', 'kill_deaths',
            'kill_assists_survived_traded', 'average_damage_per_round',
            'kills_per_round', 'assists_per_round', 'first_kills_per_round',
            'first_deaths_per_round', 'headshot_percentage', 'clutch_success_percentage'
        ]
        missing = [col for col in required_columns if col not in columns]
        if missing:
            print(f"ERROR: Missing columns in 'players' table: {missing}")
            conn.close()
            return False
        conn.close()
        return True
    except Exception as e:
        print(f"ERROR: Could not check database schema: {e}")
        return False 