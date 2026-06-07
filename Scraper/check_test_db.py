import sqlite3

def check_test_database():
    print("🔍 Checking test database schema and data...")
    
    conn = sqlite3.connect('test_clean_valorant.db')
    cursor = conn.cursor()
    
    # Check matches table schema
    print("\n📋 MATCHES TABLE COLUMNS:")
    cursor.execute("PRAGMA table_info(matches)")
    for row in cursor.fetchall():
        print(f"   {row[1]} ({row[2]})")
    
    # Check data counts
    cursor.execute("SELECT COUNT(*) FROM matches")
    matches_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM player_match_stats")
    stats_count = cursor.fetchone()[0]
    
    print(f"\n📊 DATA COUNTS:")
    print(f"   Matches: {matches_count}")
    print(f"   Player stats: {stats_count}")
    
    # Test the problematic query
    print(f"\n🧪 TESTING THE PROBLEMATIC SQL QUERY:")
    try:
        test_query = """
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
        LIMIT 5
        """
        
        cursor.execute(test_query)
        result = cursor.fetchall()
        print(f"   ✅ Query executed successfully!")
        print(f"   📊 Returned {len(result)} rows")
        
        if result:
            print(f"   🎯 Sample data:")
            for i, row in enumerate(result[:2]):
                print(f"     Row {i+1}: Player={row[0]}, Kills={row[7]}, Deaths={row[8]}")
        
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    conn.close()

if __name__ == "__main__":
    check_test_database() 