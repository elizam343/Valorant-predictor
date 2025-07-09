import sqlite3

DB_PATH = "vlr_players.db"

def list_all_players():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, team FROM players ORDER BY name")
    rows = c.fetchall()
    conn.close()
    print("\nAll Players:")
    for name, team in rows:
        print(f"{name} - {team}")

def list_all_teams():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT team FROM players ORDER BY team")
    teams = [row[0] for row in c.fetchall()]
    conn.close()
    print("\nAll Teams:")
    for team in teams:
        print(team)

def list_players_by_team():
    team = input("Enter team name: ").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM players WHERE team = ? ORDER BY name", (team,))
    players = [row[0] for row in c.fetchall()]
    conn.close()
    if players:
        print(f"\nPlayers in {team}:")
        for name in players:
            print(name)
    else:
        print(f"No players found for team '{team}'.")

def main():
    while True:
        print("\nMenu:")
        print("1. List all players")
        print("2. List all teams")
        print("3. List players by team")
        print("4. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            list_all_players()
        elif choice == "2":
            list_all_teams()
        elif choice == "3":
            list_players_by_team()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 