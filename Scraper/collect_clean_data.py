"""
Clean Data Collection System
Uses Riot API to gather validated Valorant match data
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from tqdm import tqdm

from riot_api_client import RiotAPIClient, MatchData
from clean_database_schema import CleanValorantDatabase, setup_clean_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanDataCollector:
    """Collect clean, validated Valorant data using Riot API"""
    
    def __init__(self, api_key: str, region: str = "na", db_path: str = "clean_valorant_matches.db"):
        self.api_client = RiotAPIClient(api_key, region)
        self.database = CleanValorantDatabase(db_path)
        self.collection_stats = {
            'matches_processed': 0,
            'matches_successful': 0,
            'matches_failed': 0,
            'validation_failures': 0,
            'api_errors': 0,
            'start_time': None
        }
        
    def test_setup(self) -> bool:
        """Test API connection and database"""
        logger.info("🔍 Testing setup...")
        
        # Test API connection
        if not self.api_client.test_connection():
            logger.error("❌ Riot API connection failed!")
            return False
            
        # Test database
        try:
            stats = self.database.get_database_stats()
            logger.info(f"✅ Database ready: {stats['matches']} existing matches")
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
            
        logger.info("✅ Setup test successful!")
        return True
    
    def get_seed_players(self) -> List[Dict[str, str]]:
        """Get a list of known pro players to start data collection"""
        # These are example pro players - you can expand this list
        seed_players = [
            {"name": "TenZ", "tag": "SEN"},
            {"name": "Aspas", "tag": "LOUD"},
            {"name": "Chronicle", "tag": "FNC"},
            {"name": "Demon1", "tag": "EG"},
            {"name": "mindfreak", "tag": "PRX"},
            {"name": "ScreaM", "tag": "KC"},
            {"name": "Derke", "tag": "FNC"},
            {"name": "yay", "tag": "C9"},
            {"name": "Sacy", "tag": "SEN"},
            {"name": "Marved", "tag": "NRG"},
            {"name": "crashies", "tag": "NRG"},
            {"name": "Victor", "tag": "NRG"},
            {"name": "s0m", "tag": "NRG"},
            {"name": "FNS", "tag": "NRG"},
            {"name": "Shao", "tag": "NAVI"},
            {"name": "cNed", "tag": "NAVI"},
            {"name": "SUYGETSU", "tag": "NAVI"},
            {"name": "Ange1", "tag": "NAVI"},
            {"name": "ardiis", "tag": "NAVI"},
        ]
        
        logger.info(f"📋 Using {len(seed_players)} seed players for data collection")
        return seed_players
    
    def collect_player_matches(self, player_name: str, player_tag: str, max_matches: int = 20) -> int:
        """Collect recent matches for a specific player"""
        logger.info(f"🎮 Collecting matches for {player_name}#{player_tag}")
        
        # Get player PUUID
        player_data = self.api_client.get_player_by_name(player_name, player_tag)
        if not player_data:
            logger.warning(f"⚠️ Player {player_name}#{player_tag} not found")
            return 0
            
        puuid = player_data['puuid']
        
        # Get match history
        match_ids = self.api_client.get_match_history(puuid, max_matches)
        if not match_ids:
            logger.warning(f"⚠️ No matches found for {player_name}#{player_tag}")
            return 0
        
        successful_matches = 0
        
        for match_id in tqdm(match_ids, desc=f"Processing {player_name} matches"):
            self.collection_stats['matches_processed'] += 1
            
            try:
                # Get match details
                match_data = self.api_client.get_match_details(match_id)
                
                if match_data:
                    # Insert into database
                    if self.database.insert_match_data(match_data):
                        successful_matches += 1
                        self.collection_stats['matches_successful'] += 1
                    else:
                        self.collection_stats['validation_failures'] += 1
                else:
                    self.collection_stats['api_errors'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing match {match_id}: {e}")
                self.collection_stats['matches_failed'] += 1
                
            # Small delay to be respectful to API
            time.sleep(0.5)
        
        logger.info(f"✅ Collected {successful_matches}/{len(match_ids)} matches for {player_name}")
        return successful_matches
    
    def collect_seed_data(self, matches_per_player: int = 20) -> Dict:
        """Collect initial dataset using seed players"""
        logger.info("🌱 Starting seed data collection...")
        self.collection_stats['start_time'] = datetime.now()
        
        seed_players = self.get_seed_players()
        total_collected = 0
        
        for player in tqdm(seed_players, desc="Collecting from seed players"):
            try:
                collected = self.collect_player_matches(
                    player['name'], 
                    player['tag'], 
                    matches_per_player
                )
                total_collected += collected
                
                # Progress update every few players
                if self.collection_stats['matches_processed'] % 50 == 0:
                    self.print_progress()
                    
            except Exception as e:
                logger.error(f"❌ Error collecting from {player['name']}: {e}")
                continue
        
        # Final stats
        self.print_final_stats()
        return self.collection_stats
    
    def expand_dataset(self, target_matches: int = 1000) -> Dict:
        """Expand dataset by finding more players from existing matches"""
        logger.info(f"📈 Expanding dataset to {target_matches} matches...")
        
        current_stats = self.database.get_database_stats()
        current_matches = current_stats['matches']
        
        if current_matches >= target_matches:
            logger.info(f"✅ Already have {current_matches} matches (target: {target_matches})")
            return self.collection_stats
        
        # Get unique players from existing matches
        # This would require a database query to find players we haven't fully explored
        logger.info("🔍 Finding new players from existing matches...")
        
        # For now, just collect more from seed players
        additional_needed = target_matches - current_matches
        matches_per_new_round = max(5, additional_needed // 20)
        
        return self.collect_seed_data(matches_per_new_round)
    
    def print_progress(self):
        """Print collection progress"""
        stats = self.collection_stats
        success_rate = (stats['matches_successful'] / max(stats['matches_processed'], 1)) * 100
        
        logger.info(f"📊 Progress: {stats['matches_processed']} processed, "
                   f"{stats['matches_successful']} successful ({success_rate:.1f}%)")
    
    def print_final_stats(self):
        """Print final collection statistics"""
        stats = self.collection_stats
        db_stats = self.database.get_database_stats()
        
        elapsed = datetime.now() - stats['start_time'] if stats['start_time'] else timedelta(0)
        
        print(f"\n🎉 DATA COLLECTION COMPLETE!")
        print(f"⏱️  Total time: {elapsed}")
        print(f"📊 Matches processed: {stats['matches_processed']}")
        print(f"✅ Successful: {stats['matches_successful']}")
        print(f"❌ Failed: {stats['matches_failed']}")
        print(f"⚠️  Validation failures: {stats['validation_failures']}")
        print(f"🔌 API errors: {stats['api_errors']}")
        print(f"\n📈 DATABASE STATS:")
        print(f"   Matches: {db_stats['matches']:,}")
        print(f"   Players: {db_stats['players']:,}")
        print(f"   Total records: {db_stats['total_records']:,}")
        print(f"   Avg kills per match: {db_stats['avg_kills']}")
        print(f"   Kill range: {db_stats['kill_range']}")
        
        # Data quality check
        is_clean, issues = self.database.validate_data_quality()
        if is_clean:
            print(f"✅ Data quality: EXCELLENT")
        else:
            print(f"⚠️  Data quality issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"   - {issue}")
    
    def quick_test_collection(self, test_player: str = "TenZ", test_tag: str = "SEN") -> bool:
        """Quick test to collect one player's data"""
        logger.info(f"🧪 Quick test: collecting data for {test_player}#{test_tag}")
        
        if not self.test_setup():
            return False
            
        matches_collected = self.collect_player_matches(test_player, test_tag, 5)
        
        if matches_collected > 0:
            logger.info(f"✅ Test successful! Collected {matches_collected} matches")
            
            # Show sample data
            stats = self.database.get_database_stats()
            logger.info(f"📊 Database now has: {stats}")
            
            return True
        else:
            logger.error("❌ Test failed - no matches collected")
            return False

def main():
    """Main data collection script"""
    print("🎮 VALORANT CLEAN DATA COLLECTOR")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your Riot API key: ").strip()
    if not api_key or api_key == "YOUR_RIOT_API_KEY_HERE":
        print("❌ Please provide a valid Riot API key")
        return
    
    # Get region
    region = input("Enter region (na/eu/ap/kr) [default: na]: ").strip().lower() or "na"
    
    # Setup collector
    collector = CleanDataCollector(api_key, region)
    
    print(f"\n🔧 Setup complete!")
    print(f"   Region: {region}")
    print(f"   Database: clean_valorant_matches.db")
    
    # Menu
    while True:
        print(f"\n📋 COLLECTION OPTIONS:")
        print(f"1. 🧪 Quick test (collect 5 matches from TenZ)")
        print(f"2. 🌱 Collect seed data (20 matches from 19 pro players)")
        print(f"3. 📈 Expand dataset (collect up to 1000 total matches)")
        print(f"4. 📊 Show database stats")
        print(f"5. 🔍 Validate data quality")
        print(f"6. ❌ Exit")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == "1":
            collector.quick_test_collection()
        elif choice == "2":
            collector.collect_seed_data()
        elif choice == "3":
            target = int(input("Target number of matches [1000]: ") or 1000)
            collector.expand_dataset(target)
        elif choice == "4":
            stats = collector.database.get_database_stats()
            print(f"\n📊 DATABASE STATS:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        elif choice == "5":
            is_clean, issues = collector.database.validate_data_quality()
            if is_clean:
                print("✅ Data quality is excellent!")
            else:
                print(f"⚠️ Found {len(issues)} data quality issues:")
                for issue in issues:
                    print(f"   - {issue}")
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 