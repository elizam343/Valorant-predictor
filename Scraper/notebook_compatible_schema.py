"""
Notebook-Compatible Database Schema
Creates the exact schema format that works with the existing notebook
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotebookCompatibleDatabase:
    """Database schema that exactly matches what the notebook expects"""
    
    def __init__(self, db_path: str = "clean_valorant_matches.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with notebook-compatible schema"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Drop existing tables to recreate with correct schema
            tables_to_drop = ['player_match_stats', 'matches', 'teams', 'tournaments', 'maps', 'players']
            for table in tables_to_drop:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Players table - EXACTLY like test database
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)
            
            # Teams table - EXACTLY like test database
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)
            
            # Tournaments table - EXACTLY like test database
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tournaments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                )
            """)
            
            # Maps table - EXACTLY like test database
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    map_name TEXT NOT NULL
                )
            """)
            
            # Matches table - EXACTLY like test database (WITH match_id column!)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    tournament_id INTEGER,
                    match_date TEXT,
                    series_type TEXT,
                    FOREIGN KEY (tournament_id) REFERENCES tournaments (id)
                )
            """)
            
            # Player match stats table - EXACTLY like test database
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_match_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    match_id INTEGER,
                    map_id INTEGER,
                    team_id INTEGER,
                    kills INTEGER,
                    deaths INTEGER,
                    assists INTEGER,
                    acs REAL,
                    adr REAL,
                    fk INTEGER,
                    hs_percentage REAL,
                    kdr REAL,
                    FOREIGN KEY (player_id) REFERENCES players (id),
                    FOREIGN KEY (match_id) REFERENCES matches (id),
                    FOREIGN KEY (map_id) REFERENCES maps (id),
                    FOREIGN KEY (team_id) REFERENCES teams (id)
                )
            """)
            
            conn.commit()
            logger.info("✅ Notebook-compatible database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """Get current database statistics"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            stats = {}
            
            tables = ['players', 'teams', 'tournaments', 'maps', 'matches', 'player_match_stats']
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                except:
                    stats[table] = 0
            
            return stats
        finally:
            conn.close()

def update_production_database():
    """Update the production database to use notebook-compatible schema"""
    print("🔧 UPDATING PRODUCTION DATABASE TO NOTEBOOK-COMPATIBLE SCHEMA")
    print("=" * 70)
    
    # Backup the old database first
    import shutil
    try:
        shutil.copy("clean_valorant_matches.db", "clean_valorant_matches_OLD.db")
        print("✅ Backed up old database to clean_valorant_matches_OLD.db")
    except:
        print("⚠️ No existing database to backup")
    
    # Create new compatible database
    db = NotebookCompatibleDatabase()
    stats = db.get_database_stats()
    
    print(f"\n📊 NEW DATABASE STRUCTURE:")
    for table, count in stats.items():
        print(f"   {table}: {count} records")
    
    print(f"\n✅ Production database is now notebook-compatible!")
    print(f"📍 Location: C:\\Users\\dinos\\OneDrive\\Desktop\\Valorant-predictor\\Scraper\\clean_valorant_matches.db")

if __name__ == "__main__":
    update_production_database() 