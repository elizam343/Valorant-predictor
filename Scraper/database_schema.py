"""
Database schema for Valorant match data storage
Handles matches, players, teams, and detailed statistics
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValorantDatabase:
    def __init__(self, db_path: str = "valorant_matches.db"):
        self.db_path = db_path
        self.timeout = 120.0  # Increased timeout for all connections
        self.init_database()
    
    def init_database(self):
        """Initialize the database with all required tables"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            
            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    region TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    team_id INTEGER,
                    region TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams (id)
                )
            """)
            
            # Tournaments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tournaments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    region TEXT,
                    tier TEXT,
                    start_date DATE,
                    end_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Matches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY,
                    match_id INTEGER UNIQUE NOT NULL,
                    tournament_id INTEGER,
                    team1_id INTEGER,
                    team2_id INTEGER,
                    winner_id INTEGER,
                    match_date TIMESTAMP,
                    series_type TEXT,
                    total_maps INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tournament_id) REFERENCES tournaments (id),
                    FOREIGN KEY (team1_id) REFERENCES teams (id),
                    FOREIGN KEY (team2_id) REFERENCES teams (id),
                    FOREIGN KEY (winner_id) REFERENCES teams (id)
                )
            """)
            
            # Maps table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    map_name TEXT NOT NULL,
                    map_number INTEGER,
                    team1_score INTEGER,
                    team2_score,
                    winner_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches (id),
                    FOREIGN KEY (winner_id) REFERENCES teams (id)
                )
            """)
            
            # Player match statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_match_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    map_id INTEGER,
                    player_id INTEGER,
                    team_id INTEGER,
                    agent TEXT,
                    kills INTEGER,
                    deaths INTEGER,
                    assists INTEGER,
                    acs REAL,
                    adr REAL,
                    fk INTEGER,
                    hs_percentage REAL,
                    kdr REAL,
                    rounds_played INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches (id),
                    FOREIGN KEY (map_id) REFERENCES maps (id),
                    FOREIGN KEY (player_id) REFERENCES players (id),
                    FOREIGN KEY (team_id) REFERENCES teams (id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_tournament ON matches(tournament_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_match_stats(match_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_match_stats(player_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_match_stats(team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(name)")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
        finally:
            conn.close()
    
    def get_or_create_team(self, team_name: str, region: str = None, conn=None, cursor=None) -> int:
        """Get team ID or create if doesn't exist. Uses provided conn/cursor if given."""
        own_conn = False
        if conn is None or cursor is None:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            own_conn = True
        try:
            # Handle null/empty team names
            if not team_name or team_name.strip() == "":
                team_name = "Unknown Team"
            # Try to get existing team
            cursor.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
            result = cursor.fetchone()
            if result:
                return result[0]
            # Create new team
            cursor.execute(
                "INSERT INTO teams (name, region) VALUES (?, ?)",
                (team_name, region)
            )
            if own_conn:
                conn.commit()
            return cursor.lastrowid
        finally:
            if own_conn:
                conn.close()
    
    def get_or_create_player(self, player_name: str, team_id: int = None, region: str = None, conn=None, cursor=None) -> int:
        """Get player ID or create if doesn't exist. Uses provided conn/cursor if given."""
        own_conn = False
        if conn is None or cursor is None:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            own_conn = True
        try:
            if not player_name or player_name.strip() == "":
                player_name = "Unknown Player"
            cursor.execute("SELECT id FROM players WHERE name = ?", (player_name,))
            result = cursor.fetchone()
            if result:
                player_id = result[0]
                if team_id:
                    cursor.execute(
                        "UPDATE players SET team_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (team_id, player_id)
                    )
                if own_conn:
                    conn.commit()
                return player_id
            cursor.execute(
                "INSERT INTO players (name, team_id, region) VALUES (?, ?, ?)",
                (player_name, team_id, region)
            )
            if own_conn:
                conn.commit()
            return cursor.lastrowid
        finally:
            if own_conn:
                conn.close()

    def get_or_create_tournament(self, tournament_name: str, region: str = None, tier: str = None, conn=None, cursor=None) -> int:
        """Get tournament ID or create if doesn't exist. Uses provided conn/cursor if given."""
        own_conn = False
        if conn is None or cursor is None:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            own_conn = True
        try:
            cursor.execute("SELECT id FROM tournaments WHERE name = ?", (tournament_name,))
            result = cursor.fetchone()
            if result:
                return result[0]
            cursor.execute(
                "INSERT INTO tournaments (name, region, tier) VALUES (?, ?, ?)",
                (tournament_name, region, tier)
            )
            if own_conn:
                conn.commit()
            return cursor.lastrowid
        finally:
            if own_conn:
                conn.close()

    def insert_match(self, match_data: Dict[str, Any]) -> int:
        """Insert a complete match with all related data"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        cursor = conn.cursor()
        try:
            match_id = match_data.get('match_id')
            date_str = match_data.get('date', '')
            tournament_name = match_data.get('tournament', 'Unknown Tournament')
            series_type = match_data.get('series_type', 'bo1')
            try:
                match_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except:
                match_date = datetime.now()
            # Use the same connection/cursor for tournament
            tournament_id = self.get_or_create_tournament(tournament_name, conn=conn, cursor=cursor)
            teams = set()
            for map_stat in match_data.get('map_stats', []):
                for player in map_stat.get('flat_players', []):
                    team_name = player.get('team', 'Unknown Team')
                    if team_name and team_name.strip():
                        teams.add(team_name)
            team_list = list(teams)
            if len(team_list) >= 2:
                team1_name, team2_name = team_list[0], team_list[1]
            elif len(team_list) == 1:
                team1_name, team2_name = team_list[0], 'Unknown Team 2'
            else:
                team1_name, team2_name = 'Unknown Team 1', 'Unknown Team 2'
            team1_id = self.get_or_create_team(team1_name, conn=conn, cursor=cursor)
            team2_id = self.get_or_create_team(team2_name, conn=conn, cursor=cursor)
            winner_id = None
            cursor.execute("""
                INSERT OR REPLACE INTO matches 
                (match_id, tournament_id, team1_id, team2_id, winner_id, match_date, series_type, total_maps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (match_id, tournament_id, team1_id, team2_id, winner_id, match_date, series_type, len(match_data.get('map_stats', []))))
            match_db_id = cursor.lastrowid
            for map_idx, map_stat in enumerate(match_data.get('map_stats', [])):
                map_name = map_stat.get('map', 'Unknown')
                cursor.execute("""
                    INSERT INTO maps (match_id, map_name, map_number)
                    VALUES (?, ?, ?)
                """, (match_db_id, map_name, map_idx + 1))
                map_id = cursor.lastrowid
                for player_data in map_stat.get('flat_players', []):
                    player_name = player_data.get('name', 'Unknown Player')
                    team_name = player_data.get('team', 'Unknown Team')
                    team_id = self.get_or_create_team(team_name, conn=conn, cursor=cursor)
                    player_id = self.get_or_create_player(player_name, team_id, conn=conn, cursor=cursor)
                    kills = int(player_data.get('kills', 0))
                    deaths = int(player_data.get('deaths', 0)) if player_data.get('deaths') else 0
                    assists = int(player_data.get('assists', 0)) if player_data.get('assists') else 0
                    acs_str = player_data.get('acs', '0')
                    adr_str = player_data.get('adr', '0')
                    fk_str = player_data.get('fk', '0')
                    hs_str = player_data.get('hs%', '0%')
                    kdr_str = player_data.get('kdr', '0')
                    acs = float(acs_str.split('\n')[0]) if acs_str and acs_str != '0' else 0.0
                    adr = float(adr_str.split('\n')[0]) if adr_str and adr_str != '0' else 0.0
                    fk = int(fk_str.split('\n')[0]) if fk_str and fk_str != '0' else 0
                    hs_percentage = float(hs_str.replace('%', '').split('\n')[0]) if hs_str and hs_str != '0%' else 0.0
                    kdr = float(kdr_str.split('\n')[0]) if kdr_str and kdr_str != '0' else 0.0
                    cursor.execute("""
                        INSERT INTO player_match_stats 
                        (match_id, map_id, player_id, team_id, kills, deaths, assists, acs, adr, fk, hs_percentage, kdr)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (match_db_id, map_id, player_id, team_id, kills, deaths, assists, acs, adr, fk, hs_percentage, kdr))
            conn.commit()
            logger.info(f"Successfully inserted match {match_id} with {len(match_data.get('map_stats', []))} maps")
            return match_db_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting match {match_data.get('match_id')}: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_match_count(self) -> int:
        """Get total number of matches in database"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM matches")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_player_count(self) -> int:
        """Get total number of players in database"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM players")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_team_count(self) -> int:
        """Get total number of teams in database"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM teams")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_player_stats(self, player_name: str, limit: int = 50) -> List[Dict]:
        """Get recent match statistics for a player"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    m.match_date,
                    t1.name as team_name,
                    t2.name as opponent_name,
                    pms.kills,
                    pms.deaths,
                    pms.assists,
                    pms.acs,
                    pms.adr,
                    pms.kdr,
                    mp.map_name
                FROM player_match_stats pms
                JOIN players p ON pms.player_id = p.id
                JOIN matches m ON pms.match_id = m.id
                JOIN teams t1 ON pms.team_id = t1.id
                JOIN teams t2 ON (m.team1_id = t2.id OR m.team2_id = t2.id) AND t2.id != t1.id
                JOIN maps mp ON pms.map_id = mp.id
                WHERE p.name = ?
                ORDER BY m.match_date DESC
                LIMIT ?
            """, (player_name, limit))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_recent_matches(self, limit: int = 20) -> List[Dict]:
        """Get recent matches with basic info"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    m.match_id,
                    m.match_date,
                    t1.name as team1_name,
                    t2.name as team2_name,
                    tour.name as tournament_name,
                    m.series_type,
                    m.total_maps
                FROM matches m
                JOIN teams t1 ON m.team1_id = t1.id
                JOIN teams t2 ON m.team2_id = t2.id
                JOIN tournaments tour ON m.tournament_id = tour.id
                ORDER BY m.match_date DESC
                LIMIT ?
            """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

if __name__ == "__main__":
    # Test database initialization
    db = ValorantDatabase("test_valorant.db")
    print(f"Database initialized with {db.get_match_count()} matches")
    print(f"Players: {db.get_player_count()}")
    print(f"Teams: {db.get_team_count()}") 