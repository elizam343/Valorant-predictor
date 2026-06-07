"""
Enhanced scraper that uses database storage and incremental updates
Keeps match data current and organized
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from database_schema import ValorantDatabase
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedValorantScraper:
    def __init__(self, db_path: str = "valorant_matches.db"):
        self.db = ValorantDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://www.vlr.gg"
        self.rate_limit_delay = 1.0  # seconds between requests
    
    def get_latest_match_id(self) -> int:
        """Get the latest match ID from the database"""
        with self.db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(match_id) FROM matches")
            result = cursor.fetchone()
            return result[0] if result[0] else 0
    
    def scrape_match(self, match_id: int) -> Optional[Dict]:
        """Scrape a single match from VLR.gg"""
        try:
            url = f"{self.base_url}/{match_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse match data (simplified - you'd need to implement full parsing)
            # This is a placeholder for the actual parsing logic
            match_data = self.parse_match_page(response.text, match_id)
            
            if match_data:
                return match_data
            
        except requests.RequestException as e:
            logger.error(f"Error scraping match {match_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error scraping match {match_id}: {str(e)}")
        
        return None
    
    def parse_match_page(self, html_content: str, match_id: int) -> Optional[Dict]:
        """Parse match page HTML to extract match data"""
        # This is a simplified parser - you'd need to implement full HTML parsing
        # For now, return a basic structure
        try:
            # Extract basic match info from HTML
            # This would use BeautifulSoup to parse the actual page structure
            
            match_data = {
                'match_id': match_id,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tournament': 'Unknown Tournament',
                'series_type': 'bo1',
                'map_stats': []
            }
            
            # Add placeholder map stats
            # In reality, you'd extract this from the HTML
            match_data['map_stats'].append({
                'map': 'Ascent',
                'flat_players': [
                    {
                        'name': 'Player1',
                        'team': 'Team1',
                        'kills': '15',
                        'deaths': '10',
                        'assists': '5',
                        'acs': '250',
                        'adr': '180',
                        'fk': '3',
                        'hs%': '25%',
                        'kdr': '1.5'
                    }
                ]
            })
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error parsing match {match_id}: {str(e)}")
            return None
    
    def scrape_range(self, start_id: int, end_id: int, batch_size: int = 10):
        """Scrape a range of match IDs"""
        logger.info(f"Scraping matches from {start_id} to {end_id}")
        
        successful = 0
        failed = 0
        
        for match_id in tqdm(range(start_id, end_id + 1), desc="Scraping matches"):
            try:
                # Check if match already exists
                with self.db.db_path as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM matches WHERE match_id = ?", (match_id,))
                    if cursor.fetchone()[0] > 0:
                        logger.debug(f"Match {match_id} already exists, skipping")
                        continue
                
                # Scrape match
                match_data = self.scrape_match(match_id)
                
                if match_data:
                    # Insert into database
                    self.db.insert_match(match_data)
                    successful += 1
                    logger.debug(f"Successfully scraped match {match_id}")
                else:
                    failed += 1
                    logger.debug(f"Failed to scrape match {match_id}")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Batch processing
                if (match_id - start_id + 1) % batch_size == 0:
                    logger.info(f"Processed {match_id - start_id + 1} matches: {successful} successful, {failed} failed")
                
            except Exception as e:
                failed += 1
                logger.error(f"Error processing match {match_id}: {str(e)}")
        
        logger.info(f"Scraping complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def update_recent_matches(self, days_back: int = 7):
        """Update with recent matches from the last N days"""
        logger.info(f"Updating with matches from the last {days_back} days")
        
        # Get the latest match ID we have
        latest_id = self.db.get_latest_match_id()
        
        # Estimate how many matches to check (rough estimate)
        # You might want to implement a more sophisticated approach
        estimated_matches_per_day = 50
        matches_to_check = days_back * estimated_matches_per_day
        
        end_id = latest_id + matches_to_check
        
        return self.scrape_range(latest_id + 1, end_id)
    
    def get_missing_matches(self, start_id: int, end_id: int) -> List[int]:
        """Get list of missing match IDs in a range"""
        with self.db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT match_id FROM matches 
                WHERE match_id BETWEEN ? AND ?
                ORDER BY match_id
            """, (start_id, end_id))
            
            existing_ids = set(row[0] for row in cursor.fetchall())
            all_ids = set(range(start_id, end_id + 1))
            
            return sorted(list(all_ids - existing_ids))
    
    def fill_gaps(self, start_id: int, end_id: int, max_gap_size: int = 1000):
        """Fill gaps in match data"""
        missing_ids = self.get_missing_matches(start_id, end_id)
        
        if not missing_ids:
            logger.info("No missing matches found")
            return
        
        logger.info(f"Found {len(missing_ids)} missing matches")
        
        # Process in chunks to avoid overwhelming the server
        for i in range(0, len(missing_ids), max_gap_size):
            chunk = missing_ids[i:i + max_gap_size]
            logger.info(f"Processing chunk {i//max_gap_size + 1}: {len(chunk)} matches")
            
            successful, failed = self.scrape_range(min(chunk), max(chunk))
            
            if failed > 0:
                logger.warning(f"Chunk had {failed} failures")
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        stats = {
            'matches': self.db.get_match_count(),
            'players': self.db.get_player_count(),
            'teams': self.db.get_team_count(),
            'latest_match_id': self.db.get_latest_match_id(),
            'date_range': self.get_date_range()
        }
        
        return stats
    
    def get_date_range(self) -> Dict:
        """Get the date range of matches in the database"""
        with self.db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(match_date), MAX(match_date) 
                FROM matches 
                WHERE match_date IS NOT NULL
            """)
            result = cursor.fetchone()
            
            if result and result[0] and result[1]:
                return {
                    'earliest': result[0],
                    'latest': result[1]
                }
            return {'earliest': None, 'latest': None}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove very old match data to keep database size manageable"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        logger.info(f"Cleaning up matches older than {cutoff_date}")
        
        with self.db.db_path as conn:
            cursor = conn.cursor()
            
            # Count matches to be deleted
            cursor.execute("""
                SELECT COUNT(*) FROM matches 
                WHERE match_date < ?
            """, (cutoff_date,))
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Delete old matches (cascade will handle related data)
                cursor.execute("""
                    DELETE FROM matches 
                    WHERE match_date < ?
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Deleted {count} old matches")
            else:
                logger.info("No old matches to delete")

def main():
    """Main scraper function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Valorant match scraper")
    parser.add_argument("--db-path", default="valorant_matches.db", help="Database file path")
    parser.add_argument("--start-id", type=int, help="Start match ID")
    parser.add_argument("--end-id", type=int, help="End match ID")
    parser.add_argument("--update-recent", type=int, default=7, help="Update with recent matches (days)")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill gaps in existing data")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--cleanup", type=int, help="Clean up matches older than N days")
    
    args = parser.parse_args()
    
    scraper = EnhancedValorantScraper(args.db_path)
    
    if args.stats:
        stats = scraper.get_database_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.cleanup:
        scraper.cleanup_old_data(args.cleanup)
    
    elif args.fill_gaps:
        if args.start_id and args.end_id:
            scraper.fill_gaps(args.start_id, args.end_id)
        else:
            print("Error: --fill-gaps requires --start-id and --end-id")
    
    elif args.update_recent:
        scraper.update_recent_matches(args.update_recent)
    
    elif args.start_id and args.end_id:
        scraper.scrape_range(args.start_id, args.end_id)
    
    else:
        print("Please specify an action: --stats, --cleanup, --fill-gaps, --update-recent, or --start-id/--end-id")

if __name__ == "__main__":
    main() 