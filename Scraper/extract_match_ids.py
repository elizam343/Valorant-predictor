"""
Extract valid match IDs from existing corrupted database
We'll use these IDs to re-scrape with fixed parsing logic
"""

import sqlite3
import json
from typing import List, Dict

def extract_match_ids(db_path: str = "valorant_matches_CORRUPTED_BACKUP.db") -> Dict:
    """Extract all unique match IDs from the corrupted database"""
    print("🔍 Extracting match IDs from existing database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get total unique match IDs
        cursor.execute("SELECT COUNT(DISTINCT match_id) FROM matches")
        total_matches = cursor.fetchone()[0]
        
        # Get sample of match IDs to check range
        cursor.execute("SELECT DISTINCT match_id FROM matches ORDER BY match_id LIMIT 10")
        sample_ids = [row[0] for row in cursor.fetchall()]
        
        # Get min and max match IDs
        cursor.execute("SELECT MIN(match_id), MAX(match_id) FROM matches")
        min_id, max_id = cursor.fetchone()
        
        # Get all unique match IDs 
        cursor.execute("SELECT DISTINCT match_id FROM matches ORDER BY match_id")
        all_match_ids = [row[0] for row in cursor.fetchall()]
        
        # Check for gaps (to understand ID distribution)
        gaps = []
        if len(all_match_ids) > 1:
            for i in range(min(100, len(all_match_ids) - 1)):  # Check first 100 for gaps
                gap = all_match_ids[i+1] - all_match_ids[i]
                if gap > 1:
                    gaps.append(gap)
        
        stats = {
            'total_matches': total_matches,
            'sample_ids': sample_ids,
            'min_id': min_id,
            'max_id': max_id,
            'id_range': max_id - min_id,
            'all_match_ids': all_match_ids,
            'avg_gap': sum(gaps) / len(gaps) if gaps else 1,
            'max_gap': max(gaps) if gaps else 1
        }
        
        print(f"📊 MATCH ID ANALYSIS:")
        print(f"   Total unique matches: {total_matches:,}")
        print(f"   ID range: {min_id} to {max_id}")
        print(f"   Sample IDs: {sample_ids}")
        print(f"   Average gap between IDs: {stats['avg_gap']:.1f}")
        print(f"   Max gap: {stats['max_gap']}")
        
        return stats
        
    finally:
        conn.close()

def save_match_ids_for_scraping(match_ids: List[int], output_file: str = "match_ids_to_scrape.json"):
    """Save match IDs to a file for batch scraping"""
    print(f"💾 Saving {len(match_ids):,} match IDs to {output_file}")
    
    # Split into chunks for easier processing
    chunk_size = 1000
    chunks = [match_ids[i:i + chunk_size] for i in range(0, len(match_ids), chunk_size)]
    
    data = {
        'total_matches': len(match_ids),
        'chunks': len(chunks),
        'chunk_size': chunk_size,
        'match_ids': match_ids,
        'chunks_list': chunks
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved {len(chunks)} chunks of match IDs")
    return chunks

def analyze_existing_data_quality():
    """Quick analysis of what data quality issues we had"""
    print("\n🔍 ANALYZING EXISTING DATA QUALITY ISSUES...")
    
    conn = sqlite3.connect("valorant_matches_CORRUPTED_BACKUP.db")
    cursor = conn.cursor()
    
    try:
        # Check for impossible values in existing data
        cursor.execute("SELECT COUNT(*) FROM player_match_stats WHERE kills > 100")
        extreme_kills = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_match_stats WHERE kills < 0") 
        negative_kills = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_match_stats WHERE deaths = 0")
        zero_deaths = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_match_stats")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(kills), MIN(kills), MAX(kills) FROM player_match_stats")
        avg_kills, min_kills, max_kills = cursor.fetchone()
        
        print(f"📊 DATA QUALITY ISSUES IN EXISTING DB:")
        print(f"   Total records: {total_records:,}")
        print(f"   Extreme kills (>100): {extreme_kills:,} ({100*extreme_kills/total_records:.1f}%)")
        print(f"   Negative kills: {negative_kills:,}")
        print(f"   Zero deaths: {zero_deaths:,} ({100*zero_deaths/total_records:.1f}%)")
        print(f"   Kill stats: avg={avg_kills:.1f}, min={min_kills}, max={max_kills}")
        
        return {
            'total_records': total_records,
            'extreme_kills_pct': 100*extreme_kills/total_records,
            'zero_deaths_pct': 100*zero_deaths/total_records,
            'avg_kills': avg_kills
        }
        
    finally:
        conn.close()

if __name__ == "__main__":
    print("🎮 MATCH ID EXTRACTOR FOR VLR.GG RE-SCRAPING")
    print("=" * 60)
    
    # Extract match IDs
    stats = extract_match_ids()
    
    # Analyze data quality issues  
    quality_stats = analyze_existing_data_quality()
    
    # Save match IDs for re-scraping
    chunks = save_match_ids_for_scraping(stats['all_match_ids'])
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📈 Found {stats['total_matches']:,} unique match IDs to re-scrape")
    print(f"   📦 Split into {len(chunks)} chunks of 1000 matches each")
    print(f"   🔧 This will be MUCH faster than scanning all IDs sequentially")
    print(f"   ✅ Ready to create fixed VLR.gg scraper")
    
    print(f"\n💡 BENEFITS OF TARGETED SCRAPING:")
    print(f"   • Skip {stats['id_range'] - stats['total_matches']:,} non-existent match IDs")
    print(f"   • Only scrape known valid matches")
    print(f"   • ~{stats['total_matches'] / max(1, stats['avg_gap']):.0f}x faster than sequential scanning")
    print(f"   • Get clean data with fixed parsing logic")
    
    print(f"\n🔍 NEXT STEPS:")
    print(f"   1. Create fixed VLR.gg scraper with proper parsing")
    print(f"   2. Add data validation to prevent corruption")
    print(f"   3. Re-scrape all {stats['total_matches']:,} matches with clean logic")
    print(f"   4. Expect ~15-20 avg kills per match (vs current {quality_stats['avg_kills']:.1f})") 