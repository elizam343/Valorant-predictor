"""
Monitor Database Progress While Scraper Is Running
"""

import sqlite3
import os
from datetime import datetime

def monitor_database():
    print("📊 DATABASE MONITORING")
    print("=" * 50)
    
    db_path = "clean_valorant_matches.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    # Get file size
    file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"📁 Database file: {db_path}")
    print(f"💾 File size: {file_size:.2f} MB")
    
    # Check database content
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count records in each table
        tables = ['players', 'teams', 'tournaments', 'maps', 'matches', 'player_match_stats']
        
        print(f"\n📈 RECORD COUNTS:")
        total_player_records = 0
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count:,} records")
                if table == 'player_match_stats':
                    total_player_records = count
            except Exception as e:
                print(f"   {table}: Error - {e}")
        
        # Sample recent data if any exists
        if total_player_records > 0:
            print(f"\n🎯 SAMPLE DATA (Latest 5 records):")
            cursor.execute("""
                SELECT p.name, mp.map_name, pms.kills, pms.deaths, pms.assists
                FROM player_match_stats pms
                JOIN players p ON pms.player_id = p.id
                JOIN maps mp ON pms.map_id = mp.id
                ORDER BY pms.id DESC
                LIMIT 5
            """)
            
            rows = cursor.fetchall()
            if rows:
                print("   Player | Map | K | D | A")
                print("   " + "-" * 30)
                for row in rows:
                    print(f"   {row[0][:12]:<12} | {row[1][:8]:<8} | {row[2]:2} | {row[3]:2} | {row[4]:2}")
                
                # Calculate average kills
                cursor.execute("SELECT AVG(kills) FROM player_match_stats WHERE kills > 0")
                avg_kills = cursor.fetchone()[0]
                if avg_kills:
                    print(f"\n📊 Average kills per record: {avg_kills:.1f}")
                    
                    if 10 <= avg_kills <= 30:
                        print("   ✅ Data quality looks good!")
                    else:
                        print("   ⚠️ Data quality may need review")
        else:
            print(f"\n⏳ NO DATA YET - Scraper is still starting up")
            print(f"   Wait a few minutes for data to appear")
        
        conn.close()
        
        # Check scraping progress if file exists
        if os.path.exists('scraping_progress.json'):
            import json
            try:
                with open('scraping_progress.json', 'r') as f:
                    progress = json.load(f)
                
                print(f"\n🔄 SCRAPING PROGRESS:")
                print(f"   Chunks completed: {progress.get('completed_chunks', 0)}/697")
                print(f"   Matches processed: {progress.get('stats', {}).get('completed', 0):,}")
                print(f"   Success rate: {progress.get('stats', {}).get('successful', 0):,}/{progress.get('stats', {}).get('completed', 0):,}")
                
                remaining_chunks = 697 - progress.get('completed_chunks', 0)
                estimated_hours = remaining_chunks * 0.053  # ~3.2 minutes per chunk
                print(f"   Estimated time remaining: {estimated_hours:.1f} hours")
                
            except:
                print(f"\n🔄 No progress file found")
        
        print(f"\n📍 FULL DATABASE PATH:")
        print(f"   {os.path.abspath(db_path)}")
        print(f"\n💡 Use this path for Google Colab upload!")
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    monitor_database() 