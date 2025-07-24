#!/usr/bin/env python3
"""
Simple JSON to database migrator with single connection
Avoids database locking issues
"""

import os
import json
import glob
import sqlite3
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMigrator:
    def __init__(self, json_dir: str, db_path: str):
        self.json_dir = Path(json_dir)
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def init_database(self):
        """Initialize database with single connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=60.0)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                region TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                team_id INTEGER,
                region TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams (id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                region TEXT,
                tier TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                match_id INTEGER UNIQUE NOT NULL,
                tournament_id INTEGER,
                team1_id INTEGER,
                team2_id INTEGER,
                match_date TIMESTAMP,
                series_type TEXT,
                total_maps INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tournament_id) REFERENCES tournaments (id),
                FOREIGN KEY (team1_id) REFERENCES teams (id),
                FOREIGN KEY (team2_id) REFERENCES teams (id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                map_name TEXT NOT NULL,
                map_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                map_id INTEGER,
                player_id INTEGER,
                team_id INTEGER,
                kills INTEGER,
                deaths INTEGER,
                assists INTEGER,
                acs REAL,
                adr REAL,
                fk INTEGER,
                hs_percentage REAL,
                kdr REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (id),
                FOREIGN KEY (map_id) REFERENCES maps (id),
                FOREIGN KEY (player_id) REFERENCES players (id),
                FOREIGN KEY (team_id) REFERENCES teams (id)
            )
        """)
        
        # Create indexes
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_match_stats(match_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(name)")
        
        self.conn.commit()
        logger.info("Database initialized successfully")
    
    def get_or_create_team(self, team_name: str) -> int:
        """Get or create team"""
        if not team_name or team_name.strip() == "":
            team_name = "Unknown Team"
        
        self.cursor.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO teams (name) VALUES (?)", (team_name,))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_or_create_player(self, player_name: str, team_id: int) -> int:
        """Get or create player"""
        if not player_name or player_name.strip() == "":
            player_name = "Unknown Player"
        
        self.cursor.execute("SELECT id FROM players WHERE name = ?", (player_name,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO players (name, team_id) VALUES (?, ?)", (player_name, team_id))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_or_create_tournament(self, tournament_name: str) -> int:
        """Get or create tournament"""
        self.cursor.execute("SELECT id FROM tournaments WHERE name = ?", (tournament_name,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO tournaments (name) VALUES (?)", (tournament_name,))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_match(self, match_data: dict) -> bool:
        """Insert a match with all related data"""
        try:
            match_id = match_data.get('match_id')
            date_str = match_data.get('date', '')
            tournament_name = match_data.get('tournament', 'Unknown Tournament')
            series_type = match_data.get('series_type', 'bo1')
            
            # Parse date
            try:
                match_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except:
                match_date = datetime.now()
            
            # Get or create tournament
            tournament_id = self.get_or_create_tournament(tournament_name)
            
            # Get teams from map stats
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
            
            # Get or create teams
            team1_id = self.get_or_create_team(team1_name)
            team2_id = self.get_or_create_team(team2_name)
            
            # Insert match
            self.cursor.execute("""
                INSERT OR REPLACE INTO matches 
                (match_id, tournament_id, team1_id, team2_id, match_date, series_type, total_maps)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (match_id, tournament_id, team1_id, team2_id, match_date, series_type, len(match_data.get('map_stats', []))))
            
            match_db_id = self.cursor.lastrowid
            
            # Insert maps and player stats
            for map_idx, map_stat in enumerate(match_data.get('map_stats', [])):
                map_name = map_stat.get('map', 'Unknown')
                
                # Insert map
                self.cursor.execute("""
                    INSERT INTO maps (match_id, map_name, map_number)
                    VALUES (?, ?, ?)
                """, (match_db_id, map_name, map_idx + 1))
                
                map_id = self.cursor.lastrowid
                
                # Insert player statistics
                for player_data in map_stat.get('flat_players', []):
                    player_name = player_data.get('name', 'Unknown Player')
                    team_name = player_data.get('team', 'Unknown Team')
                    
                    # Get or create player and team
                    team_id = self.get_or_create_team(team_name)
                    player_id = self.get_or_create_player(player_name, team_id)
                    
                    # Parse statistics
                    kills = int(player_data.get('kills', 0))
                    deaths = int(player_data.get('deaths', 0)) if player_data.get('deaths') else 0
                    assists = int(player_data.get('assists', 0)) if player_data.get('assists') else 0
                    
                    # Parse complex stats
                    acs_str = player_data.get('acs', '0')
                    adr_str = player_data.get('adr', '0')
                    fk_str = player_data.get('fk', '0')
                    hs_str = player_data.get('hs%', '0%')
                    kdr_str = player_data.get('kdr', '0')
                    
                    # Extract first value from multi-line stats
                    acs = float(acs_str.split('\n')[0]) if acs_str and acs_str != '0' else 0.0
                    adr = float(adr_str.split('\n')[0]) if adr_str and adr_str != '0' else 0.0
                    fk = int(fk_str.split('\n')[0]) if fk_str and fk_str != '0' else 0
                    hs_percentage = float(hs_str.replace('%', '').split('\n')[0]) if hs_str and hs_str != '0%' else 0.0
                    kdr = float(kdr_str.split('\n')[0]) if kdr_str and kdr_str != '0' else 0.0
                    
                    # Insert player match stats
                    self.cursor.execute("""
                        INSERT INTO player_match_stats 
                        (match_id, map_id, player_id, team_id, kills, deaths, assists, acs, adr, fk, hs_percentage, kdr)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (match_db_id, map_id, player_id, team_id, kills, deaths, assists, acs, adr, fk, hs_percentage, kdr))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting match {match_data.get('match_id')}: {str(e)}")
            self.conn.rollback()
            return False
    
    def migrate_files(self, batch_size: int = 100):
        """Migrate all JSON files"""
        # Get all JSON files
        pattern = str(self.json_dir / "match_*.json")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} JSON files to migrate")
        
        if len(files) == 0:
            logger.warning("No JSON files found")
            return
        
        # Initialize database
        self.init_database()
        
        processed = 0
        errors = 0
        
        try:
            # Process files in batches
            for i in tqdm(range(0, len(files), batch_size), desc="Migrating batches"):
                batch_files = files[i:i + batch_size]
                
                for file_path in batch_files:
                    try:
                        # Load JSON data
                        with open(file_path, 'r', encoding='utf-8') as f:
                            match_data = json.load(f)
                        
                        # Add match_id if not present
                        if 'match_id' not in match_data:
                            match_id = int(Path(file_path).stem.split('_')[-1])
                            match_data['match_id'] = match_id
                        
                        # Insert into database
                        if self.insert_match(match_data):
                            processed += 1
                        else:
                            errors += 1
                            
                    except Exception as e:
                        errors += 1
                        logger.warning(f"Error processing {file_path}: {str(e)}")
                
                # Log progress
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"Processed {processed} files, {errors} errors")
        
        finally:
            # Close connection
            if self.conn:
                self.conn.close()
        
        # Print summary
        logger.info("=" * 50)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {processed}")
        logger.info(f"Errors encountered: {errors}")
        logger.info(f"Success rate: {processed / (processed + errors) * 100:.1f}%")
        
        # Get final stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        matches = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM players")
        players = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM teams")
        teams = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"Matches in database: {matches}")
        logger.info(f"Players in database: {players}")
        logger.info(f"Teams in database: {teams}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple JSON to database migration")
    parser.add_argument("--json-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--db-path", required=True, help="Database file path")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    migrator = SimpleMigrator(args.json_dir, args.db_path)
    migrator.migrate_files(args.batch_size)

if __name__ == "__main__":
    main() 