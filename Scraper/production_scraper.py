"""
Production VLR.gg Scraper for Bulk Re-scraping
Re-scrapes all 69,691 match IDs with clean parsing logic
"""

import json
import time
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from fixed_vlr_scraper import FixedVLRScraper, ValidatedPlayerStats
from notebook_compatible_schema import NotebookCompatibleDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionScraper:
    """Production scraper for bulk re-scraping with database integration"""
    
    def __init__(self, db_path: str = "clean_valorant_matches.db"):
        self.scraper = FixedVLRScraper(db_path)
        self.database = NotebookCompatibleDatabase(db_path)
        
        # Production settings
        self.scraper.rate_limit = 2.0  # Slower rate for bulk scraping
        
        # Statistics
        self.production_stats = {
            'total_to_scrape': 0,
            'completed': 0,
            'successful': 0,
            'failed': 0,
            'validation_failures': 0,
            'duplicate_skips': 0,
            'start_time': None,
            'estimated_completion': None
        }
    
    def load_match_ids(self, filename: str = "match_ids_to_scrape.json") -> List[int]:
        """Load match IDs from our extracted list"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return data['match_ids']
        except FileNotFoundError:
            logger.error(f"❌ Match IDs file {filename} not found!")
            logger.error("   Run extract_match_ids.py first to create this file")
            return []
    
    def convert_match_to_database_format(self, match_data: Dict) -> bool:
        """Convert scraped match data to database format and insert - FIXED VERSION"""
        try:
            import sqlite3
            from datetime import datetime
            
            # Connect to the database
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            match_id = match_data['match_id']
            teams = match_data.get('teams', ['Team A', 'Team B'])
            match_date = match_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            tournament = match_data.get('tournament', 'Unknown Tournament')
            
            # Insert teams if they don't exist
            team_ids = {}
            for i, team_name in enumerate(teams):
                cursor.execute("INSERT OR IGNORE INTO teams (name) VALUES (?)", (team_name,))
                cursor.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
                team_ids[team_name] = cursor.fetchone()[0]
            
            # Insert tournament if it doesn't exist
            cursor.execute("INSERT OR IGNORE INTO tournaments (name) VALUES (?)", (tournament,))
            cursor.execute("SELECT id FROM tournaments WHERE name = ?", (tournament,))
            tournament_id = cursor.fetchone()[0]
            
            # Insert match
            cursor.execute("""
                INSERT OR IGNORE INTO matches (match_id, tournament_id, match_date, series_type)
                VALUES (?, ?, ?, ?)
            """, (match_id, tournament_id, match_date, 'bo3'))  # Default to bo3
            
            # Get the match row id
            cursor.execute("SELECT id FROM matches WHERE match_id = ?", (match_id,))
            match_row_id = cursor.fetchone()[0]
            
            total_players_inserted = 0
            
            # Process each map
            for map_data in match_data.get('map_stats', []):
                map_name = map_data.get('map_name', 'Unknown')
                
                # Insert map if it doesn't exist
                cursor.execute("INSERT OR IGNORE INTO maps (map_name) VALUES (?)", (map_name,))
                cursor.execute("SELECT id FROM maps WHERE map_name = ?", (map_name,))
                map_id = cursor.fetchone()[0]
                
                # Process each player in this map
                for player in map_data.get('players', []):
                    # Insert player if they don't exist
                    cursor.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (player.name,))
                    cursor.execute("SELECT id FROM players WHERE name = ?", (player.name,))
                    player_id = cursor.fetchone()[0]
                    
                    # Get team_id
                    team_id = team_ids.get(player.team, team_ids.get(teams[0], 1))
                    
                    # Insert player match stats - SAME FORMAT AS TEST DATABASE
                    cursor.execute("""
                        INSERT OR IGNORE INTO player_match_stats 
                        (player_id, match_id, map_id, team_id, kills, deaths, assists, acs, adr, fk, hs_percentage, kdr)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id, match_row_id, map_id, team_id,
                        player.kills, player.deaths, player.assists,
                        player.acs, player.adr, player.first_kills,
                        player.headshot_percentage, player.kdr
                    ))
                    
                    total_players_inserted += 1
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Match {match_id}: {len(match_data.get('map_stats', []))} maps, {total_players_inserted} players SAVED TO DATABASE")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database conversion error for match {match_data.get('match_id', 'unknown')}: {e}")
            return False
    
    def scrape_chunk(self, match_ids: List[int], chunk_number: int, total_chunks: int) -> Dict:
        """Scrape a chunk of matches"""
        print(f"\n🎮 PROCESSING CHUNK {chunk_number}/{total_chunks}")
        print(f"📊 Matches in this chunk: {len(match_ids)}")
        
        chunk_stats = {
            'successful': 0,
            'failed': 0,
            'validation_failures': 0
        }
        
        for i, match_id in enumerate(tqdm(match_ids, desc=f"Chunk {chunk_number}")):
            self.production_stats['completed'] += 1
            
            try:
                # Scrape the match
                match_data = self.scraper.scrape_match(match_id)
                
                if match_data:
                    # Convert and save to database
                    if self.convert_match_to_database_format(match_data):
                        chunk_stats['successful'] += 1
                        self.production_stats['successful'] += 1
                    else:
                        chunk_stats['failed'] += 1
                        self.production_stats['failed'] += 1
                else:
                    chunk_stats['failed'] += 1
                    self.production_stats['failed'] += 1
                
                # Update validation failures from scraper
                self.production_stats['validation_failures'] += self.scraper.stats.get('validation_failures', 0)
                
                # Progress update every 50 matches
                if (i + 1) % 50 == 0:
                    self._print_progress_update(chunk_number, i + 1, len(match_ids))
                
            except Exception as e:
                logger.error(f"❌ Error processing match {match_id}: {e}")
                chunk_stats['failed'] += 1
                self.production_stats['failed'] += 1
        
        return chunk_stats
    
    def _print_progress_update(self, chunk_num: int, chunk_progress: int, chunk_size: int):
        """Print progress update"""
        overall_progress = (self.production_stats['completed'] / self.production_stats['total_to_scrape']) * 100
        success_rate = (self.production_stats['successful'] / max(self.production_stats['completed'], 1)) * 100
        
        # Estimate time remaining
        if self.production_stats['start_time']:
            elapsed = datetime.now() - self.production_stats['start_time']
            rate = self.production_stats['completed'] / elapsed.total_seconds()
            remaining_matches = self.production_stats['total_to_scrape'] - self.production_stats['completed']
            remaining_time = remaining_matches / rate if rate > 0 else 0
            
            print(f"📊 Progress: {overall_progress:.1f}% | "
                  f"Success: {success_rate:.1f}% | "
                  f"ETA: {remaining_time/3600:.1f}h")
    
    def full_production_scrape(self, start_chunk: int = 0, max_chunks: int = None) -> Dict:
        """Run full production scraping of all matches"""
        print("🏭 STARTING FULL PRODUCTION SCRAPING")
        print("=" * 60)
        
        # Load match IDs
        match_ids = self.load_match_ids()
        if not match_ids:
            return {'error': 'No match IDs loaded'}
        
        # Split into chunks
        chunk_size = 100  # Smaller chunks for better progress tracking
        total_chunks = (len(match_ids) + chunk_size - 1) // chunk_size
        
        if max_chunks:
            total_chunks = min(total_chunks, start_chunk + max_chunks)
        
        self.production_stats['total_to_scrape'] = len(match_ids[start_chunk * chunk_size:total_chunks * chunk_size])
        self.production_stats['start_time'] = datetime.now()
        
        print(f"📈 Total matches to scrape: {self.production_stats['total_to_scrape']:,}")
        print(f"📦 Chunks to process: {total_chunks - start_chunk}")
        print(f"🎯 Expected time: {self.production_stats['total_to_scrape'] * 2.5 / 3600:.1f} hours")
        
        # Process chunks
        for chunk_idx in range(start_chunk, total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(match_ids))
            chunk_match_ids = match_ids[start_idx:end_idx]
            
            chunk_stats = self.scrape_chunk(chunk_match_ids, chunk_idx + 1, total_chunks)
            
            # Save progress after each chunk
            self._save_progress_checkpoint(chunk_idx + 1)
            
            print(f"✅ Chunk {chunk_idx + 1} complete: "
                  f"{chunk_stats['successful']}/{len(chunk_match_ids)} successful")
        
        self._print_final_production_stats()
        return self.production_stats
    
    def _save_progress_checkpoint(self, completed_chunks: int):
        """Save progress checkpoint"""
        checkpoint = {
            'completed_chunks': completed_chunks,
            'stats': self.production_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('scraping_progress.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _print_final_production_stats(self):
        """Print final production statistics"""
        elapsed = datetime.now() - self.production_stats['start_time'] if self.production_stats['start_time'] else 0
        
        print(f"\n🎉 PRODUCTION SCRAPING COMPLETED!")
        print(f"⏱️  Total time: {elapsed}")
        print(f"📊 Total processed: {self.production_stats['completed']:,}")
        print(f"✅ Successful: {self.production_stats['successful']:,}")
        print(f"❌ Failed: {self.production_stats['failed']:,}")
        print(f"⚠️  Validation failures: {self.production_stats['validation_failures']:,}")
        
        success_rate = (self.production_stats['successful'] / max(self.production_stats['completed'], 1)) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if self.production_stats['successful'] > 0:
            print(f"\n🎯 ESTIMATED DATA QUALITY:")
            estimated_players = self.production_stats['successful'] * 10  # ~10 players per match
            print(f"   Total clean player records: ~{estimated_players:,}")
            print(f"   Expected average kills: 12-18 per match")
            print(f"   Expected average deaths: 12-18 per match")
            print(f"   Expected average assists: 3-8 per match")

def main():
    """Main production scraping interface"""
    print("🏭 VALORANT PRODUCTION SCRAPER")
    print("=" * 60)
    print("This will re-scrape all 69,691 matches with clean parsing logic")
    
    scraper = ProductionScraper()
    
    while True:
        print(f"\n📋 PRODUCTION OPTIONS:")
        print(f"1. 🧪 Test run (scrape first 100 matches)")
        print(f"2. 🏭 Full production run (all 69,691 matches)")
        print(f"3. 🔄 Resume from checkpoint")
        print(f"4. 📊 Show current database stats")
        print(f"5. ❌ Exit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == "1":
            print("\n🧪 STARTING TEST RUN...")
            results = scraper.full_production_scrape(start_chunk=0, max_chunks=1)
            print("✅ Test run completed!")
            
        elif choice == "2":
            confirm = input("\n⚠️ This will take 8-12 hours. Continue? (yes/no): ").strip().lower()
            if confirm == 'yes':
                print("\n🏭 STARTING FULL PRODUCTION RUN...")
                results = scraper.full_production_scrape()
            else:
                print("❌ Cancelled")
                
        elif choice == "3":
            try:
                with open('scraping_progress.json', 'r') as f:
                    checkpoint = json.load(f)
                start_chunk = checkpoint['completed_chunks']
                print(f"\n🔄 Resuming from chunk {start_chunk}...")
                results = scraper.full_production_scrape(start_chunk=start_chunk)
            except FileNotFoundError:
                print("❌ No checkpoint file found")
                
        elif choice == "4":
            stats = scraper.database.get_database_stats()
            print(f"\n📊 CURRENT DATABASE STATS:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
                
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 