"""
Migration script to convert JSON match files to SQLite database
Handles bulk conversion of all scraped match data
"""

import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import logging
from database_schema import ValorantDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONToDatabaseMigrator:
    def __init__(self, json_dir: str = "../scraped_matches", db_path: str = "valorant_matches.db"):
        self.json_dir = Path(json_dir)
        self.db = ValorantDatabase(db_path)
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    def get_json_files(self) -> list:
        """Get all JSON match files"""
        pattern = str(self.json_dir / "match_*.json")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} JSON files to migrate")
        return sorted(files)
    
    def extract_match_id_from_filename(self, filename: str) -> int:
        """Extract match ID from filename like 'match_169058.json'"""
        try:
            return int(filename.split('_')[-1].replace('.json', ''))
        except:
            return None
    
    def load_json_file(self, filepath: str) -> dict:
        """Load and parse JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add match_id if not present
            if 'match_id' not in data:
                match_id = self.extract_match_id_from_filename(os.path.basename(filepath))
                if match_id:
                    data['match_id'] = match_id
            
            return data
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def migrate_all_files(self, batch_size: int = 100, start_from: int = None):
        """Migrate all JSON files to database"""
        files = self.get_json_files()
        
        if start_from:
            # Filter files to start from specific match ID
            files = [f for f in files if self.extract_match_id_from_filename(f) >= start_from]
            logger.info(f"Starting migration from match ID {start_from}, {len(files)} files remaining")
        
        logger.info(f"Starting migration of {len(files)} files...")
        
        # Process files in batches
        for i in tqdm(range(0, len(files), batch_size), desc="Migrating batches"):
            batch_files = files[i:i + batch_size]
            self.migrate_batch(batch_files)
            
            # Log progress
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {self.processed_count} files, {self.error_count} errors")
        
        self.print_summary()
    
    def migrate_batch(self, file_paths: list):
        """Migrate a batch of files"""
        for file_path in file_paths:
            try:
                # Load JSON data
                match_data = self.load_json_file(file_path)
                if not match_data:
                    self.error_count += 1
                    continue
                
                # Insert into database
                self.db.insert_match(match_data)
                self.processed_count += 1
                
            except Exception as e:
                self.error_count += 1
                self.errors.append({
                    'file': file_path,
                    'error': str(e)
                })
                logger.warning(f"Error processing {file_path}: {str(e)} - skipping file")
                continue  # Skip this file and continue with the next one
    
    def print_summary(self):
        """Print migration summary"""
        logger.info("=" * 50)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {self.processed_count}")
        logger.info(f"Errors encountered: {self.error_count}")
        logger.info(f"Success rate: {self.processed_count / (self.processed_count + self.error_count) * 100:.1f}%")
        
        # Database stats
        logger.info(f"Matches in database: {self.db.get_match_count()}")
        logger.info(f"Players in database: {self.db.get_player_count()}")
        logger.info(f"Teams in database: {self.db.get_team_count()}")
        
        if self.errors:
            logger.info("\nTop 10 errors:")
            for error in self.errors[:10]:
                logger.info(f"  {error['file']}: {error['error']}")
    
    def verify_migration(self, sample_size: int = 10):
        """Verify migration by checking sample files"""
        logger.info("Verifying migration...")
        
        files = self.get_json_files()[:sample_size]
        verified_count = 0
        
        for file_path in files:
            try:
                # Load original JSON
                original_data = self.load_json_file(file_path)
                if not original_data:
                    continue
                
                match_id = original_data.get('match_id')
                if not match_id:
                    continue
                
                # Check if match exists in database
                with self.db.db_path as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM matches WHERE match_id = ?", (match_id,))
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        verified_count += 1
                    else:
                        logger.warning(f"Match {match_id} not found in database")
                
            except Exception as e:
                logger.error(f"Error verifying {file_path}: {str(e)}")
        
        logger.info(f"Verification complete: {verified_count}/{len(files)} files verified")
    
    def cleanup_old_files(self, backup_dir: str = None):
        """Optionally backup and remove old JSON files"""
        if not backup_dir:
            backup_dir = str(self.json_dir / "backup")
        
        logger.info(f"Backing up JSON files to {backup_dir}")
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        files = self.get_json_files()
        
        for file_path in tqdm(files, desc="Backing up files"):
            try:
                filename = os.path.basename(file_path)
                backup_path = os.path.join(backup_dir, filename)
                
                # Move file to backup
                os.rename(file_path, backup_path)
                
            except Exception as e:
                logger.error(f"Error backing up {file_path}: {str(e)}")
        
        logger.info(f"Backup complete. {len(files)} files moved to {backup_dir}")

def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate JSON match files to SQLite database")
    parser.add_argument("--json-dir", default="../scraped_matches", help="Directory containing JSON files")
    parser.add_argument("--db-path", default="valorant_matches.db", help="Database file path")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--start-from", type=int, help="Start migration from specific match ID")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    parser.add_argument("--cleanup", action="store_true", help="Backup and remove JSON files after migration")
    parser.add_argument("--backup-dir", help="Backup directory for JSON files")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = JSONToDatabaseMigrator(args.json_dir, args.db_path)
    
    # Run migration
    migrator.migrate_all_files(args.batch_size, args.start_from)
    
    # Verify if requested
    if args.verify:
        migrator.verify_migration()
    
    # Cleanup if requested
    if args.cleanup:
        migrator.cleanup_old_files(args.backup_dir)

if __name__ == "__main__":
    main() 