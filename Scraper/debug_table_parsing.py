"""
Debug script to test our table parsing logic step by step
"""

import requests
from bs4 import BeautifulSoup
import json

def debug_table_parsing():
    """Debug our exact parsing logic"""
    print("🔍 DEBUGGING TABLE PARSING STEP BY STEP")
    print("=" * 60)
    
    # Use a known working match ID
    match_id = 14510
    url = f"https://www.vlr.gg/{match_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(f"✅ Loaded match {match_id}")
    
    # Step 1: Find map containers (like our scraper does)
    print(f"\n🎯 STEP 1: Finding map containers")
    map_containers = soup.select('.vm-stats-game')
    print(f"   Found {len(map_containers)} map containers")
    
    for map_idx, map_container in enumerate(map_containers):
        print(f"\n📍 MAP {map_idx + 1}:")
        
        # Step 2: Find tables in this map container (like our scraper does)
        team_tables = map_container.select('.wf-table-inset table')
        print(f"   Tables found: {len(team_tables)}")
        
        if len(team_tables) == 0:
            # Try alternative selector
            team_tables = map_container.select('table')
            print(f"   Alternative tables found: {len(team_tables)}")
        
        for team_idx, table in enumerate(team_tables[:2]):
            print(f"\n   🏆 TEAM {team_idx + 1} TABLE:")
            
            # Step 3: Find player rows (like our scraper does)
            player_rows = table.select('tbody tr')
            print(f"      Player rows found: {len(player_rows)}")
            
            if len(player_rows) == 0:
                # Try alternative selector
                player_rows = table.select('tr')
                print(f"      Alternative rows found: {len(player_rows)}")
            
            # Step 4: Parse each player row
            for row_idx, row in enumerate(player_rows[:6]):  # Limit to 6 for debug
                print(f"\n      👤 ROW {row_idx + 1}:")
                
                cells = row.find_all('td')
                print(f"         Cells found: {len(cells)}")
                
                if len(cells) >= 4:
                    # Try to extract player name
                    name_cell = cells[0].get_text(strip=True)
                    print(f"         Name cell: '{name_cell}'")
                    
                    # Try to extract kills (column 4)
                    if len(cells) > 4:
                        kills_cell = cells[4]
                        kills_html = str(kills_cell)
                        kills_text = kills_cell.get_text()
                        
                        print(f"         Kills cell HTML: {kills_html[:200]}...")
                        print(f"         Kills cell text: '{kills_text}'")
                        
                        # Test our mod-both extraction
                        mod_both_span = kills_cell.find('span', class_='side mod-both') or kills_cell.find('span', class_='side mod-side mod-both')
                        if mod_both_span:
                            mod_both_value = mod_both_span.get_text(strip=True)
                            print(f"         ✅ Found mod-both kills: '{mod_both_value}'")
                        else:
                            print(f"         ❌ No mod-both span found")
                            # Check what spans are there
                            all_spans = kills_cell.find_all('span')
                            print(f"         All spans: {[span.get('class') for span in all_spans]}")
                else:
                    print(f"         ❌ Not enough cells ({len(cells)}) - might be header row")

if __name__ == "__main__":
    debug_table_parsing() 