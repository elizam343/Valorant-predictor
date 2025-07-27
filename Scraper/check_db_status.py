"""
Quick script to check current database status
"""

import sqlite3
import json

def check_database_status():
    print("📊 CLEAN DATABASE STATUS CHECK")
    print("=" * 50)
    
    try:
        # Check database connection
        conn = sqlite3.connect('clean_valorant_matches.db')
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Tables found: {', '.join(tables)}")
        
        # Check record counts
        if 'players' in tables:
            cursor.execute("SELECT COUNT(*) FROM players")
            players_count = cursor.fetchone()[0]
            print(f"👥 Players: {players_count:,}")
        
        if 'matches' in tables:
            cursor.execute("SELECT COUNT(*) FROM matches")
            matches_count = cursor.fetchone()[0]
            print(f"🎮 Matches: {matches_count:,}")
        
        if 'player_match_stats' in tables:
            cursor.execute("SELECT COUNT(*) FROM player_match_stats")
            stats_count = cursor.fetchone()[0]
            print(f"📈 Player records: {stats_count:,}")
            
            # Check data quality
            if stats_count > 0:
                cursor.execute("SELECT AVG(kills), MIN(kills), MAX(kills) FROM player_match_stats")
                avg_kills, min_kills, max_kills = cursor.fetchone()
                print(f"\n📊 KILL STATISTICS:")
                print(f"   Average: {avg_kills:.1f} kills")
                print(f"   Range: {min_kills} - {max_kills} kills")
                
                # Check if data looks realistic
                if 10 <= avg_kills <= 25:
                    print(f"   ✅ Data looks realistic for Valorant!")
                elif avg_kills > 50:
                    print(f"   ❌ Data still looks corrupted (too high)")
                else:
                    print(f"   ⚠️ Data looks unusual")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

def check_progress():
    print(f"\n🔄 SCRAPING PROGRESS")
    print("=" * 50)
    
    try:
        with open('scraping_progress.json', 'r') as f:
            progress = json.load(f)
        
        stats = progress['stats']
        completed_chunks = progress['completed_chunks']
        
        print(f"📦 Chunks completed: {completed_chunks}/697")
        print(f"📊 Matches processed: {stats['completed']:,}")
        print(f"✅ Successful: {stats['successful']:,}")
        print(f"❌ Failed: {stats['failed']:,}")
        print(f"⚠️ Validation failures: {stats['validation_failures']:,}")
        
        # Calculate progress
        progress_pct = (stats['completed'] / stats['total_to_scrape']) * 100
        success_rate = (stats['successful'] / max(stats['completed'], 1)) * 100
        
        print(f"\n📈 Overall progress: {progress_pct:.1f}%")
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        # Estimate time remaining
        if completed_chunks > 0:
            chunks_remaining = 697 - completed_chunks
            time_per_chunk = 3.5  # minutes per chunk based on observed rate
            hours_remaining = (chunks_remaining * time_per_chunk) / 60
            print(f"⏱️ Estimated time remaining: {hours_remaining:.1f} hours")
        
    except FileNotFoundError:
        print("❌ No progress file found")
    except Exception as e:
        print(f"❌ Error reading progress: {e}")

if __name__ == "__main__":
    check_database_status()
    check_progress()
    
    print(f"\n📍 DATABASE LOCATION:")
    import os
    db_path = os.path.abspath('clean_valorant_matches.db')
    print(f"   {db_path}")
    print(f"\n💡 Use this path for Google Colab upload!") 