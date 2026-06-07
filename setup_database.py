#!/usr/bin/env python3
"""
Comprehensive database setup and migration script
Converts JSON files to SQLite database and sets up the new system
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import sqlite3

# Add Scraper to path
sys.path.append('Scraper')

from Scraper.database_schema import ValorantDatabase
from Scraper.migrate_json_to_db import JSONToDatabaseMigrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_players_table_columns(db_path):
    required_columns = [
        ('team', 'TEXT'),
        ('rating', 'REAL'),
        ('average_combat_score', 'REAL'),
        ('kill_deaths', 'REAL'),
        ('kill_assists_survived_traded', 'REAL'),
        ('average_damage_per_round', 'REAL'),
        ('kills_per_round', 'REAL'),
        ('assists_per_round', 'REAL'),
        ('first_kills_per_round', 'REAL'),
        ('first_deaths_per_round', 'REAL'),
        ('headshot_percentage', 'REAL'),
        ('clutch_success_percentage', 'REAL'),
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(players);")
    existing_columns = [row[1] for row in cursor.fetchall()]
    for col, col_type in required_columns:
        if col not in existing_columns:
            print(f"Adding missing column '{col}' to players table...")
            cursor.execute(f"ALTER TABLE players ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()
    print("Players table schema is up to date.")

def setup_database(json_dir: str = "scraped_matches", db_path: str = "Scraper/valorant_matches.db"):
    """Set up the database and migrate JSON files"""
    logger.info("Setting up Valorant match database...")
    
    # Create Scraper directory if it doesn't exist
    os.makedirs("Scraper", exist_ok=True)
    
    # Initialize database
    logger.info("Initializing database schema...")
    db = ValorantDatabase(db_path)
    
    # Check if database already has data
    match_count = db.get_match_count()
    if match_count > 0:
        logger.info(f"Database already contains {match_count} matches")
        response = input("Do you want to re-migrate all data? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping migration. Database setup complete.")
            return db
    
    # Migrate JSON files
    logger.info("Starting JSON to database migration...")
    migrator = JSONToDatabaseMigrator(json_dir, db_path)
    
    # Get file count
    json_files = migrator.get_json_files()
    logger.info(f"Found {len(json_files)} JSON files to migrate")
    
    if len(json_files) == 0:
        logger.warning("No JSON files found. Database will be empty.")
        return db
    
    # Run migration
    try:
        migrator.migrate_all_files(batch_size=100)
        logger.info("Migration completed successfully!")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    
    return db

def verify_database(db_path: str = "Scraper/valorant_matches.db"):
    """Verify database integrity and show statistics"""
    logger.info("Verifying database...")
    
    db = ValorantDatabase(db_path)
    
    # Get statistics
    stats = {
        'matches': db.get_match_count(),
        'players': db.get_player_count(),
        'teams': db.get_team_count()
    }
    
    logger.info("Database Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:,}")
    
    # Test queries
    try:
        recent_matches = db.get_recent_matches(5)
        logger.info(f"Recent matches: {len(recent_matches)} found")
        
        if recent_matches:
            sample_player = recent_matches[0]
            logger.info(f"Sample match: {sample_player}")
        
        # Test player stats
        player_stats = db.get_player_stats("aspas", 5)
        logger.info(f"Player stats test: {len(player_stats)} records for 'aspas'")
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False
    
    return True

def test_ml_integration():
    """Test ML model integration with database"""
    logger.info("Testing ML model integration...")
    
    try:
        from kill_prediction_model.database_data_loader import DatabaseDataLoader
        
        loader = DatabaseDataLoader()
        
        # Test data loading
        df = loader.load_player_match_data(min_matches=3, days_back=365)
        logger.info(f"Loaded {len(df)} player-match records")
        
        if len(df) > 0:
            # Test feature calculation
            feature_df = loader.calculate_player_features(df)
            logger.info(f"Calculated features for {len(feature_df)} records")
            
            # Test training data preparation
            X, y, feature_columns = loader.prepare_training_data(feature_df)
            logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Test player context
            context = loader.get_player_context(
                player_name="aspas",
                team_name="MIBR",
                opponent_team="FUR",
                tournament="VCT Champions",
                series_type="bo3",
                maps=["Ascent", "Haven"]
            )
            
            if context is not None:
                logger.info(f"Player context features: {context.shape}")
            
            logger.info("ML integration test passed!")
            return True
        else:
            logger.warning("No data available for ML testing")
            return False
            
    except Exception as e:
        logger.error(f"ML integration test failed: {str(e)}")
        return False

def cleanup_old_files(json_dir: str = "scraped_matches", backup: bool = True):
    """Clean up old JSON files after successful migration"""
    if not backup:
        logger.info("Skipping file cleanup (backup disabled)")
        return
    
    logger.info("Cleaning up old JSON files...")
    
    json_path = Path(json_dir)
    if not json_path.exists():
        logger.info("No JSON directory found to clean up")
        return
    
    # Count files
    json_files = list(json_path.glob("match_*.json"))
    logger.info(f"Found {len(json_files)} JSON files to clean up")
    
    if len(json_files) == 0:
        logger.info("No JSON files to clean up")
        return
    
    # Create backup directory
    backup_dir = json_path / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    # Move files to backup
    moved_count = 0
    for file_path in json_files:
        try:
            backup_path = backup_dir / file_path.name
            file_path.rename(backup_path)
            moved_count += 1
        except Exception as e:
            logger.error(f"Error backing up {file_path}: {str(e)}")
    
    logger.info(f"Backed up {moved_count} files to {backup_dir}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up Valorant match database")
    parser.add_argument("--json-dir", default="scraped_matches", help="Directory containing JSON files")
    parser.add_argument("--db-path", default="Scraper/valorant_matches.db", help="Database file path")
    parser.add_argument("--skip-migration", action="store_true", help="Skip JSON migration")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip JSON file cleanup")
    parser.add_argument("--test-ml", action="store_true", help="Test ML model integration")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing database")
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            # Only verify existing database
            success = verify_database(args.db_path)
            if success:
                logger.info("Database verification completed successfully!")
            else:
                logger.error("Database verification failed!")
                sys.exit(1)
        else:
            # Full setup process
            logger.info("Starting Valorant database setup...")
            
            # Step 1: Set up database
            if not args.skip_migration:
                db = setup_database(args.json_dir, args.db_path)
            else:
                logger.info("Skipping migration as requested")
                db = ValorantDatabase(args.db_path)
            
            # Step 2: Verify database
            success = verify_database(args.db_path)
            if not success:
                logger.error("Database verification failed!")
                sys.exit(1)
            
            # Step 3: Test ML integration
            if args.test_ml:
                ml_success = test_ml_integration()
                if not ml_success:
                    logger.warning("ML integration test failed, but continuing...")
            
            # Step 4: Clean up old files
            if not args.skip_cleanup:
                cleanup_old_files(args.json_dir, backup=True)
            
            logger.info("Database setup completed successfully!")
            
            # Print summary
            stats = {
                'matches': db.get_match_count(),
                'players': db.get_player_count(),
                'teams': db.get_team_count()
            }
            
            logger.info("Final Database Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:,}")
            
            logger.info("\nNext steps:")
            logger.info("1. Train the model: cd kill_prediction_model && python gpu_trainer.py")
            logger.info("2. Test predictions: python advanced_matchup_predictor.py")
            logger.info("3. Run web app: cd web_app && python app.py")
            
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    db_path = "Scraper/valorant_matches.db"
    ensure_players_table_columns(db_path)
    main() 