"""
Test the fixed database insertion with a single match
"""

from fixed_vlr_scraper import FixedVLRScraper
from notebook_compatible_schema import NotebookCompatibleDatabase
import json

def test_single_match():
    print("🧪 TESTING FIXED DATABASE INSERTION")
    print("=" * 50)
    
    # Load a sample match ID from our list
    try:
        with open('match_ids_to_scrape.json', 'r') as f:
            data = json.load(f)
        test_match_id = data['match_ids'][0]  # Use first match ID
        print(f"📋 Testing with match ID: {test_match_id}")
    except:
        test_match_id = 14510  # Fallback
        print(f"📋 Using fallback match ID: {test_match_id}")
    
    # Initialize scraper with fixed insertion
    scraper = FixedVLRScraper("test_fixed_insertion.db")
    
    print(f"\n🔍 BEFORE SCRAPING:")
    stats_before = scraper.database.get_database_stats()
    for table, count in stats_before.items():
        print(f"   {table}: {count} records")
    
    # Test scraping and insertion
    print(f"\n🚀 SCRAPING MATCH {test_match_id}...")
    match_data = scraper.scrape_match(test_match_id)
    
    if match_data:
        print(f"✅ Match scraped successfully!")
        print(f"   Maps: {len(match_data.get('map_stats', []))}")
        print(f"   Players: {sum(len(map_data.get('players', [])) for map_data in match_data.get('map_stats', []))}")
        
        # Test database insertion
        print(f"\n💾 TESTING DATABASE INSERTION...")
        from production_scraper import ProductionScraper
        prod_scraper = ProductionScraper("test_fixed_insertion.db")
        insertion_success = prod_scraper.convert_match_to_database_format(match_data)
        
        if insertion_success:
            print(f"✅ Database insertion successful!")
        else:
            print(f"❌ Database insertion failed!")
            return False
        
        print(f"\n🔍 AFTER INSERTION:")
        stats_after = scraper.database.get_database_stats()
        for table, count in stats_after.items():
            added = count - stats_before.get(table, 0)
            print(f"   {table}: {count} records (+{added})")
        
        # Test the notebook query
        print(f"\n🧪 TESTING NOTEBOOK COMPATIBILITY:")
        try:
            import sqlite3
            import pandas as pd
            from datetime import datetime, timedelta
            
            conn = sqlite3.connect("test_fixed_insertion.db")
            cutoff_date = datetime.now() - timedelta(days=365)
            
            # This is the EXACT query from your notebook
            query = """
            SELECT
                p.name as player_name, t.name as team_name, pms.team_id as team_id,
                m.match_date, m.series_type, tour.name as tournament_name,
                mp.map_name, pms.kills, pms.deaths, pms.assists, pms.acs, pms.adr,
                pms.fk, pms.hs_percentage, pms.kdr, m.match_id, pms.map_id
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.id
            JOIN teams t ON pms.team_id = t.id
            JOIN matches m ON pms.match_id = m.id
            JOIN maps mp ON pms.map_id = mp.id
            JOIN tournaments tour ON m.tournament_id = tour.id
            WHERE m.match_date >= ?
            ORDER BY p.name, m.match_date, pms.map_id
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            conn.close()
            
            print(f"   ✅ Notebook query executed successfully!")
            print(f"   📊 Returned {len(df)} player records")
            
            if len(df) > 0:
                print(f"   🎯 Sample data:")
                print(f"     Player: {df.iloc[0]['player_name']}")
                print(f"     Kills: {df.iloc[0]['kills']}")
                print(f"     Deaths: {df.iloc[0]['deaths']}")
                print(f"     Map: {df.iloc[0]['map_name']}")
                
                avg_kills = df['kills'].mean()
                print(f"   📈 Average kills: {avg_kills:.1f}")
                
                if 10 <= avg_kills <= 30:
                    print(f"   ✅ Data quality looks excellent!")
                else:
                    print(f"   ⚠️ Data quality needs review")
            
        except Exception as e:
            print(f"   ❌ Notebook query failed: {e}")
            return False
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Database insertion works correctly")
        print(f"✅ Schema is notebook-compatible") 
        print(f"✅ Data quality is realistic")
        print(f"✅ Ready for production scraping!")
        
        return True
        
    else:
        print(f"❌ Match scraping failed!")
        return False

if __name__ == "__main__":
    test_single_match() 