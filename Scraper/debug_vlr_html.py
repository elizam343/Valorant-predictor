"""
Debug script to inspect VLR.gg HTML structure
Helps us understand the current page layout to fix the scraper
"""

import requests
import json
from bs4 import BeautifulSoup
import re

def inspect_match_page(match_id: int):
    """Inspect the HTML structure of a VLR.gg match page"""
    print(f"🔍 INSPECTING VLR.gg MATCH {match_id}")
    print("=" * 60)
    
    url = f"https://www.vlr.gg/{match_id}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print(f"✅ Successfully loaded page for match {match_id}")
        print(f"📄 Page title: {soup.title.get_text() if soup.title else 'No title'}")
        
        # Check if it's actually a match page
        if "vs" not in soup.get_text().lower() and "match" not in soup.get_text().lower():
            print(f"⚠️ This might not be a valid match page")
            print(f"📝 First 500 chars: {soup.get_text()[:500]}")
            return
        
        # Look for different possible selectors for match stats
        print(f"\n🎯 SEARCHING FOR MATCH STATS CONTAINERS:")
        
        # Try various selectors
        selectors_to_try = [
            '.vm-stats-game',
            '.match-stats',
            '.stats-container',
            '.scoreboard',
            '.map-stats',
            '.match-info-item',
            '[class*="stats"]',
            '[class*="scoreboard"]',
            '[class*="match"]'
        ]
        
        found_containers = []
        for selector in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                found_containers.append((selector, len(elements)))
                print(f"   ✅ {selector}: {len(elements)} elements")
            else:
                print(f"   ❌ {selector}: 0 elements")
        
        # Look for tables
        print(f"\n📊 SEARCHING FOR TABLES:")
        tables = soup.find_all('table')
        print(f"   Total tables found: {len(tables)}")
        
        for i, table in enumerate(tables[:5]):  # Check first 5 tables
            rows = table.find_all('tr')
            print(f"   Table {i+1}: {len(rows)} rows")
            
            # Check if this looks like player stats
            if len(rows) >= 5:  # At least 5 rows (header + 4+ players)
                first_row_cells = rows[0].find_all(['th', 'td'])
                print(f"     First row: {len(first_row_cells)} columns")
                if len(first_row_cells) >= 8:  # Looks like stats table
                    print(f"     ⭐ POTENTIAL STATS TABLE!")
                    # Print column headers
                    headers = [cell.get_text(strip=True) for cell in first_row_cells]
                    print(f"     Headers: {headers}")
        
        # Look for player names
        print(f"\n👥 SEARCHING FOR PLAYER NAMES:")
        player_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized names
            r'\b\w{3,15}\b'      # 3-15 character usernames
        ]
        
        text_content = soup.get_text()
        
        # Look for common Valorant player names
        common_valorant_names = ['TenZ', 'Aspas', 'Chronicle', 'ScreaM', 'Derke', 'yay', 'Sacy']
        found_names = []
        for name in common_valorant_names:
            if name in text_content:
                found_names.append(name)
        
        if found_names:
            print(f"   ✅ Found known players: {found_names}")
        else:
            print(f"   ❌ No known player names found")
        
        # Look for numbers that could be kills/deaths
        print(f"\n🔢 SEARCHING FOR STAT NUMBERS:")
        number_pattern = r'\b\d{1,2}\b'  # 1-2 digit numbers
        numbers = re.findall(number_pattern, text_content)
        number_counts = {}
        for num in numbers:
            number_counts[num] = number_counts.get(num, 0) + 1
        
        # Numbers that commonly appear in kill counts
        common_kill_numbers = ['15', '18', '20', '22', '25']
        found_kill_numbers = [num for num in common_kill_numbers if num in number_counts]
        
        if found_kill_numbers:
            print(f"   ✅ Found potential kill numbers: {found_kill_numbers}")
        else:
            print(f"   ❌ No typical kill numbers found")
        
        # Try to find specific divs/containers
        print(f"\n🏗️ CHECKING PAGE STRUCTURE:")
        
        # Look for match header
        match_header = soup.select_one('.match-header')
        if match_header:
            print(f"   ✅ Found .match-header")
            team_names = match_header.select('.team-name, .match-team-name, [class*="team"]')
            print(f"   Teams found: {len(team_names)}")
            for i, team in enumerate(team_names[:2]):
                print(f"     Team {i+1}: {team.get_text(strip=True)}")
        
        # Look for map containers
        map_containers = soup.select('.map, [class*="map"]')
        print(f"   Map-related elements: {len(map_containers)}")
        
        # Try to save HTML snippet for manual inspection
        if found_containers:
            print(f"\n💾 SAVING HTML SAMPLE:")
            best_selector = found_containers[0][0]
            sample_element = soup.select_one(best_selector)
            if sample_element:
                with open(f'vlr_html_sample_{match_id}.html', 'w', encoding='utf-8') as f:
                    f.write(str(sample_element.prettify()))
                print(f"   Saved sample HTML to vlr_html_sample_{match_id}.html")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if not found_containers:
            print(f"   ❌ No stats containers found - page structure may have changed significantly")
            print(f"   💡 Try inspecting the page manually in browser")
        elif len(tables) == 0:
            print(f"   ❌ No tables found - VLR.gg might not use tables for stats anymore")
            print(f"   💡 Look for div-based layouts instead")
        else:
            print(f"   ✅ Found {len(tables)} tables and {len(found_containers)} potential containers")
            print(f"   💡 Update CSS selectors based on findings above")
        
    except Exception as e:
        print(f"❌ Error inspecting page: {e}")

def test_multiple_matches():
    """Test multiple match IDs to find a working one"""
    # Load match IDs from our extracted list
    try:
        with open('match_ids_to_scrape.json', 'r') as f:
            data = json.load(f)
            match_ids = data['match_ids'][:10]  # Test first 10
    except FileNotFoundError:
        print("❌ No match IDs file found. Using fallback IDs")
        match_ids = [169056, 169057, 169058, 150000, 160000]
    
    print("🔍 TESTING MULTIPLE MATCH IDs")
    print("=" * 60)
    
    for match_id in match_ids:
        print(f"\n🎮 Testing match {match_id}...")
        
        url = f"https://www.vlr.gg/{match_id}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {match_id}: Valid page (200 OK)")
                
                # Quick check if it contains match data
                if "vs" in response.text.lower() or "map" in response.text.lower():
                    print(f"   🎯 {match_id}: Contains match data - GOOD CANDIDATE")
                    inspect_match_page(match_id)
                    break  # Found a good one, inspect it
                else:
                    print(f"   ⚠️ {match_id}: No match data found")
            else:
                print(f"   ❌ {match_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {match_id}: Error - {e}")

if __name__ == "__main__":
    print("🐛 VLR.GG HTML STRUCTURE DEBUGGER")
    print("=" * 60)
    print("This script helps us understand why the scraper isn't finding player data")
    print("")
    
    test_multiple_matches() 