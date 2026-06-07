"""
Clean Database Schema for Riot API Valorant Data
Designed with proper validation and constraints
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import asdict
from riot_api_client import MatchData, PlayerMatchStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanValorantDatabase:
    """Clean database with proper validation for Riot API data"""
    
    def __init__(self, db_path: str = "clean_valorant_matches.db"):
        self.db_path = db_path
        self.timeout = 120.0
        self.init_database()
    
    def init_database(self):
        """Initialize database with clean schema and validation"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Players table (simplified - using PUUID as primary key)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    puuid TEXT PRIMARY KEY,
                    player_name TEXT NOT NULL,
                    region TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_matches INTEGER DEFAULT 0
                )
            """)
            
            # Maps table (reference data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    map_name TEXT PRIMARY KEY,
                    map_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Agents table (reference data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_name TEXT PRIMARY KEY,
                    agent_role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Matches table (clean structure)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    game_start TIMESTAMP NOT NULL,
                    game_length INTEGER NOT NULL CHECK (game_length > 0),
                    map_name TEXT NOT NULL,
                    game_mode TEXT NOT NULL,
                    is_ranked BOOLEAN NOT NULL DEFAULT 0,
                    season_id TEXT,
                    rounds_played INTEGER NOT NULL CHECK (rounds_played BETWEEN 5 AND 30),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (map_name) REFERENCES maps (map_name)
                )
            """)
            
            # Team results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    won BOOLEAN NOT NULL,
                    rounds_won INTEGER NOT NULL CHECK (rounds_won >= 0),
                    rounds_lost INTEGER NOT NULL CHECK (rounds_lost >= 0),
                    FOREIGN KEY (match_id) REFERENCES matches (match_id),
                    UNIQUE(match_id, team_id)
                )
            """)
            
            # Player match statistics (the core table for ML)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_match_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    player_puuid TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    kills INTEGER NOT NULL CHECK (kills BETWEEN 0 AND 50),
                    deaths INTEGER NOT NULL CHECK (deaths BETWEEN 0 AND 50),
                    assists INTEGER NOT NULL CHECK (assists BETWEEN 0 AND 50),
                    score INTEGER NOT NULL CHECK (score >= 0),
                    damage INTEGER NOT NULL CHECK (damage >= 0),
                    headshots INTEGER NOT NULL CHECK (headshots >= 0),
                    first_kills INTEGER NOT NULL CHECK (first_kills BETWEEN 0 AND 10),
                    first_deaths INTEGER NOT NULL CHECK (first_deaths BETWEEN 0 AND 10),
                    rounds_played INTEGER NOT NULL CHECK (rounds_played BETWEEN 5 AND 30),
                    acs REAL GENERATED ALWAYS AS (CAST(score AS REAL) / rounds_played) STORED,
                    kdr REAL GENERATED ALWAYS AS (
                        CASE 
                            WHEN deaths = 0 THEN CAST(kills AS REAL)
                            ELSE CAST(kills AS REAL) / deaths 
                        END
                    ) STORED,
                    adr REAL GENERATED ALWAYS AS (CAST(damage AS REAL) / rounds_played) STORED,
                    hs_percentage REAL GENERATED ALWAYS AS (
                        CASE 
                            WHEN kills = 0 THEN 0.0
                            ELSE (CAST(headshots AS REAL) / kills) * 100 
                        END
                    ) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (match_id) REFERENCES matches (match_id),
                    FOREIGN KEY (player_puuid) REFERENCES players (puuid),
                    FOREIGN KEY (agent) REFERENCES agents (agent_name),
                    UNIQUE(match_id, player_puuid),
                    
                    -- Additional validation constraints
                    CHECK (headshots <= kills),
                    CHECK (score >= kills * 50),  -- Minimum score per kill
                    CHECK (damage >= kills * 50)  -- Minimum damage per kill
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(game_start)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_map ON matches(map_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_mode ON matches(game_mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_match_stats(player_puuid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_match_stats(match_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_date ON player_match_stats(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(player_name)")
            
            # Insert reference data
            self._insert_reference_data(cursor)
            
            conn.commit()
            logger.info("✅ Clean database schema initialized successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Database initialization failed: {e}")
            raise
        finally:
            conn.close()
    
    def _insert_reference_data(self, cursor):
        """Insert reference data for maps and agents"""
        
        # Valorant maps
        maps = [
            ('Ascent', 'Standard'),
            ('Bind', 'Standard'),
            ('Breeze', 'Standard'),
            ('Fracture', 'Standard'),
            ('Haven', 'Standard'),
            ('Icebox', 'Standard'),
            ('Lotus', 'Standard'),
            ('Pearl', 'Standard'),
            ('Split', 'Standard'),
            ('Sunset', 'Standard'),
            ('Abyss', 'Standard')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO maps (map_name, map_type) 
            VALUES (?, ?)
        """, maps)
        
        # Valorant agents
        agents = [
            ('Astra', 'Controller'),
            ('Breach', 'Initiator'),
            ('Brimstone', 'Controller'),
            ('Chamber', 'Sentinel'),
            ('Clove', 'Controller'),
            ('Cypher', 'Sentinel'),
            ('Deadlock', 'Sentinel'),
            ('Fade', 'Initiator'),
            ('Gekko', 'Initiator'),
            ('Harbor', 'Controller'),
            ('Iso', 'Duelist'),
            ('Jett', 'Duelist'),
            ('Killjoy', 'Sentinel'),
            ('Neon', 'Duelist'),
            ('Omen', 'Controller'),
            ('Phoenix', 'Duelist'),
            ('Raze', 'Duelist'),
            ('Reyna', 'Duelist'),
            ('Sage', 'Sentinel'),
            ('Skye', 'Initiator'),
            ('Sova', 'Initiator'),
            ('Viper', 'Controller'),
            ('Yoru', 'Duelist')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO agents (agent_name, agent_role) 
            VALUES (?, ?)
        """, agents)
    
    def insert_match_data(self, match_data: MatchData) -> bool:
        """Insert validated match data into database"""
        
        # Validate data first
        is_valid, error_msg = match_data.validate()
        if not is_valid:
            logger.error(f"❌ Match validation failed: {error_msg}")
            return False
        
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            
            # Insert match
            cursor.execute("""
                INSERT OR REPLACE INTO matches 
                (match_id, game_start, game_length, map_name, game_mode, 
                 is_ranked, season_id, rounds_played)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_data.match_id,
                match_data.game_start,
                match_data.game_length,
                match_data.map_name,
                match_data.game_mode,
                match_data.is_ranked,
                match_data.season_id,
                match_data.rounds_played
            ))
            
            # Insert team results
            for team_stat in match_data.team_stats:
                cursor.execute("""
                    INSERT OR REPLACE INTO team_results 
                    (match_id, team_id, won, rounds_won, rounds_lost)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    match_data.match_id,
                    team_stat['team_id'],
                    team_stat['won'],
                    team_stat['rounds_won'],
                    team_stat['rounds_lost']
                ))
            
            # Insert players and their stats
            for player_stat in match_data.player_stats:
                # Insert/update player
                cursor.execute("""
                    INSERT OR REPLACE INTO players 
                    (puuid, player_name, last_seen, total_matches)
                    VALUES (?, ?, ?, 
                        COALESCE((SELECT total_matches FROM players WHERE puuid = ?) + 1, 1)
                    )
                """, (
                    player_stat.player_puuid,
                    player_stat.player_name,
                    match_data.game_start,
                    player_stat.player_puuid
                ))
                
                # Insert player match stats
                cursor.execute("""
                    INSERT OR REPLACE INTO player_match_stats 
                    (match_id, player_puuid, player_name, team_id, agent,
                     kills, deaths, assists, score, damage, headshots,
                     first_kills, first_deaths, rounds_played)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_data.match_id,
                    player_stat.player_puuid,
                    player_stat.player_name,
                    player_stat.team_id,
                    player_stat.agent,
                    player_stat.kills,
                    player_stat.deaths,
                    player_stat.assists,
                    player_stat.score,
                    player_stat.damage,
                    player_stat.headshots,
                    player_stat.first_kills,
                    player_stat.first_deaths,
                    player_stat.rounds_played
                ))
            
            conn.commit()
            logger.info(f"✅ Successfully inserted match {match_data.match_id}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Error inserting match {match_data.match_id}: {e}")
            return False
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM matches")
            match_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM players")
            player_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM player_match_stats")
            stats_count = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("""
                SELECT MIN(game_start), MAX(game_start) 
                FROM matches
            """)
            date_range = cursor.fetchone()
            
            # Get data quality stats
            cursor.execute("""
                SELECT 
                    AVG(kills) as avg_kills,
                    AVG(deaths) as avg_deaths,
                    AVG(assists) as avg_assists,
                    AVG(acs) as avg_acs,
                    MIN(kills) as min_kills,
                    MAX(kills) as max_kills
                FROM player_match_stats
            """)
            quality_stats = cursor.fetchone()
            
            return {
                'matches': match_count,
                'players': player_count,
                'total_records': stats_count,
                'earliest_match': date_range[0],
                'latest_match': date_range[1],
                'avg_kills': round(quality_stats[0], 2) if quality_stats[0] else 0,
                'avg_deaths': round(quality_stats[1], 2) if quality_stats[1] else 0,
                'avg_assists': round(quality_stats[2], 2) if quality_stats[2] else 0,
                'avg_acs': round(quality_stats[3], 2) if quality_stats[3] else 0,
                'kill_range': f"{quality_stats[4]}-{quality_stats[5]}" if quality_stats[4] else "No data"
            }
            
        finally:
            conn.close()
    
    def validate_data_quality(self) -> Tuple[bool, List[str]]:
        """Validate data quality in the database"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        issues = []
        
        try:
            cursor = conn.cursor()
            
            # Check for impossible values
            cursor.execute("""
                SELECT COUNT(*) FROM player_match_stats 
                WHERE kills < 0 OR kills > 50
            """)
            bad_kills = cursor.fetchone()[0]
            if bad_kills > 0:
                issues.append(f"Found {bad_kills} records with impossible kill counts")
            
            cursor.execute("""
                SELECT COUNT(*) FROM player_match_stats 
                WHERE deaths = 0 AND rounds_played > 10
            """)
            zero_deaths = cursor.fetchone()[0]
            if zero_deaths > 0:
                issues.append(f"Found {zero_deaths} records with suspicious zero deaths")
            
            cursor.execute("""
                SELECT COUNT(*) FROM player_match_stats 
                WHERE headshots > kills
            """)
            bad_headshots = cursor.fetchone()[0]
            if bad_headshots > 0:
                issues.append(f"Found {bad_headshots} records with more headshots than kills")
            
            # Check average stats are reasonable
            cursor.execute("SELECT AVG(kills) FROM player_match_stats")
            avg_kills = cursor.fetchone()[0]
            if avg_kills and (avg_kills < 10 or avg_kills > 25):
                issues.append(f"Suspicious average kills: {avg_kills:.1f} (expected 10-25)")
            
            return len(issues) == 0, issues
            
        finally:
            conn.close()

# Quick setup function
def setup_clean_database(db_path: str = "clean_valorant_matches.db") -> CleanValorantDatabase:
    """Set up a new clean database"""
    logger.info(f"🗄️ Setting up clean database: {db_path}")
    db = CleanValorantDatabase(db_path)
    
    # Test the database
    stats = db.get_database_stats()
    logger.info(f"✅ Database ready - {stats['matches']} matches, {stats['players']} players")
    
    return db

if __name__ == "__main__":
    # Example usage
    db = setup_clean_database()
    
    # Show initial stats
    stats = db.get_database_stats()
    print(f"📊 Database Stats: {stats}")
    
    # Validate data quality
    is_clean, issues = db.validate_data_quality()
    if is_clean:
        print("✅ Database data quality is good!")
    else:
        print(f"⚠️ Data quality issues: {issues}") 