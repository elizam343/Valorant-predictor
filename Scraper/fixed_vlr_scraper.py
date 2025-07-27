"""
Fixed VLR.gg Scraper with Proper Parsing Logic
Addresses the original parsing bugs that caused data corruption
"""

import requests
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
from notebook_compatible_schema import NotebookCompatibleDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidatedPlayerStats:
    """Validated player statistics with proper constraints"""
    name: str
    team: str
    agent: str
    kills: int
    deaths: int
    assists: int
    acs: float
    adr: float
    first_kills: int
    headshot_percentage: float
    kdr: float
    
    def validate(self) -> Tuple[bool, str]:
        """Validate that stats are reasonable for Valorant (updated for competitive matches)"""
        if not (0 <= self.kills <= 80):  # Allow for long BO5/overtime matches
            return False, f"Invalid kills: {self.kills}"
        if not (1 <= self.deaths <= 80):  # Allow for long matches  
            return False, f"Invalid deaths: {self.deaths}"
        if not (0 <= self.assists <= 60):  # Proportional to kills
            return False, f"Invalid assists: {self.assists}"
        if not (0 <= self.acs <= 600):  # Allow for exceptional performances
            return False, f"Invalid ACS: {self.acs}"
        if not (0 <= self.adr <= 400):  # Allow for high damage games
            return False, f"Invalid ADR: {self.adr}"
        if not (0 <= self.first_kills <= 25):  # Proportional to longer matches
            return False, f"Invalid first kills: {self.first_kills}"
        if not (0 <= self.headshot_percentage <= 100):
            return False, f"Invalid HS%: {self.headshot_percentage}"
        if self.deaths > 0 and not (0 <= self.kdr <= 15):  # Allow for exceptional KDRs
            return False, f"Invalid KDR: {self.kdr}"
            
        return True, "Valid"

class FixedVLRScraper:
    """Fixed VLR.gg scraper with proper parsing and validation"""
    
    def __init__(self, db_path: str = "clean_valorant_matches.db"):
        self.base_url = "https://www.vlr.gg"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.database = NotebookCompatibleDatabase(db_path)
        self.rate_limit = 1.5  # seconds between requests
        self.last_request_time = 0
        
        # Statistics tracking
        self.stats = {
            'matches_processed': 0,
            'matches_successful': 0,
            'matches_failed': 0,
            'validation_failures': 0,
            'parsing_errors': 0,
            'start_time': None
        }
    
    def _rate_limit_wait(self):
        """Respect rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _safe_get_page(self, match_id: int) -> Optional[BeautifulSoup]:
        """Safely get a match page with error handling"""
        self._rate_limit_wait()
        
        try:
            url = f"{self.base_url}/{match_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 404:
                logger.debug(f"Match {match_id} not found (404)")
                return None
            elif response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for match {match_id}")
                return None
                
            return BeautifulSoup(response.text, 'html.parser')
            
        except requests.RequestException as e:
            logger.error(f"Request error for match {match_id}: {e}")
            return None
    
    def _clean_numeric_value(self, value: str, default: float = 0.0) -> float:
        """Clean and convert numeric values from HTML"""
        if not value:
            return default
            
        # Remove common HTML artifacts and whitespace
        cleaned = re.sub(r'[^\d.-]', '', str(value).strip())
        
        try:
            return float(cleaned) if cleaned else default
        except ValueError:
            return default
    
    def _parse_percentage(self, value: str) -> float:
        """Parse percentage values like '75%' -> 75.0"""
        if not value:
            return 0.0
        cleaned = re.sub(r'[^\d.]', '', str(value))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _extract_mod_both_value(self, cell_text: str, default: float = 0.0) -> int:
        """
        Extract the 'mod-both' value from VLR.gg stat cells
        
        VLR.gg shows stats like:
        <span class="side mod-both">23</span>    <- Overall (what we want)
        <span class="side mod-t">12</span>       <- T-side
        <span class="side mod-ct">11</span>      <- CT-side
        """
        if not cell_text:
            return int(default)
        
        # Look for mod-both pattern
        from bs4 import BeautifulSoup
        try:
            # Parse the HTML in this cell
            soup = BeautifulSoup(cell_text, 'html.parser')
            
            # Find the mod-both span
            mod_both_span = soup.find('span', class_='side mod-both') or soup.find('span', class_='side mod-side mod-both')
            
            if mod_both_span:
                value_text = mod_both_span.get_text(strip=True)
                value = self._clean_numeric_value(value_text, default)
                return int(value) if value != default else int(default)
            
            # Fallback: try to extract any number from the text
            import re
            numbers = re.findall(r'\b\d+\b', cell_text)
            if numbers:
                # Take the first reasonable number
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num <= 100:  # Reasonable range
                        return num
            
            return int(default)
            
        except Exception as e:
            logger.debug(f"Error extracting mod-both value from '{cell_text}': {e}")
            return int(default)
    
    def _extract_single_map_kills(self, kills_cell_text: str, map_index: int = 0) -> int:
        """
        Extract actual per-map kills from VLR.gg multi-line cell
        
        This was the ROOT CAUSE of the original bug!
        VLR.gg shows data like: "15\\n12\\n18" representing different rounds/segments
        The old scraper incorrectly took cumulative values
        """
        if not kills_cell_text:
            return 0
        
        # Split by newlines and clean
        parts = [part.strip() for part in str(kills_cell_text).split('\n') if part.strip()]
        
        if not parts:
            return 0
        
        # Try to get the first reasonable value (not cumulative)
        for part in parts:
            try:
                kills = int(self._clean_numeric_value(part))
                # Validate it's a reasonable per-map kill count
                if 0 <= kills <= 50:
                    return kills
            except (ValueError, TypeError):
                continue
        
        # If no valid value found, try the first part
        try:
            kills = int(self._clean_numeric_value(parts[0]))
            # Even if it's outside normal range, cap it
            return max(0, min(kills, 50))
        except (ValueError, TypeError, IndexError):
            return 0
    
    def _parse_player_stats(self, player_row, team_name: str) -> Optional[ValidatedPlayerStats]:
        """Parse a single player's statistics from HTML row"""
        try:
            cells = player_row.find_all('td')
            if len(cells) < 8:  # Need at least 8 columns
                return None
            
            # Extract basic info
            name_cell = cells[0].get_text(strip=True)
            agent_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            
            # Extract stats - FIXED COLUMN MAPPING based on debug results
            # Headers: ['', '', 'R2.0', 'ACS', 'K', 'D', 'A', '+/–', 'KAST', 'ADR', 'HS%', 'FK', 'FD', '+/–']
            #           0    1    2      3     4   5   6   7     8      9     10     11    12    13
            acs_raw = cells[3].get_text('\n', strip=True) if len(cells) > 3 else "0"      # ACS
            kills_raw = cells[4].get_text('\n', strip=True) if len(cells) > 4 else "0"    # K
            deaths_raw = cells[5].get_text('\n', strip=True) if len(cells) > 5 else "1"   # D  
            assists_raw = cells[6].get_text('\n', strip=True) if len(cells) > 6 else "0"  # A
            adr_raw = cells[9].get_text('\n', strip=True) if len(cells) > 9 else "0"      # ADR
            hs_raw = cells[10].get_text('\n', strip=True) if len(cells) > 10 else "0%"    # HS%
            fk_raw = cells[11].get_text('\n', strip=True) if len(cells) > 11 else "0"     # FK
            # KDR will be calculated from kills/deaths
            kdr_raw = "0"
            
            # FIXED PARSING - Extract the "mod-both" values (overall stats)
            kills = self._extract_mod_both_value(kills_raw)
            deaths = max(1, self._extract_mod_both_value(deaths_raw, 1))  # Deaths must be at least 1
            assists = self._extract_mod_both_value(assists_raw)
            acs = self._extract_mod_both_value(acs_raw)
            adr = self._extract_mod_both_value(adr_raw) 
            first_kills = self._extract_mod_both_value(fk_raw)
            headshot_pct = self._extract_mod_both_value(hs_raw)
            kdr = self._clean_numeric_value(kdr_raw)
            
            # Calculate KDR if not provided
            if kdr == 0 and deaths > 0:
                kdr = kills / deaths
            
            player_stats = ValidatedPlayerStats(
                name=name_cell,
                team=team_name,
                agent=agent_cell,
                kills=kills,
                deaths=deaths,
                assists=assists,
                acs=acs,
                adr=adr,
                first_kills=first_kills,
                headshot_percentage=headshot_pct,
                kdr=kdr
            )
            
            return player_stats
            
        except Exception as e:
            logger.error(f"Error parsing player stats: {e}")
            return None
    
    def _extract_match_info(self, soup: BeautifulSoup, match_id: int) -> Optional[Dict]:
        """Extract basic match information"""
        try:
            match_info = {'match_id': match_id}
            
            # Extract teams
            team_elements = soup.select('.match-header-vs .match-header-vs-team-name')
            if len(team_elements) >= 2:
                match_info['teams'] = [t.get_text(strip=True) for t in team_elements[:2]]
            else:
                match_info['teams'] = ['Team A', 'Team B']
            
            # Extract date
            date_element = soup.select_one('.moment-tz-convert')
            if date_element and date_element.get('data-utc-ts'):
                match_info['date'] = date_element.get('data-utc-ts')
            else:
                match_info['date'] = datetime.now().isoformat()
            
            # Extract tournament
            tournament_element = soup.select_one('.match-header-event-series')
            match_info['tournament'] = tournament_element.get_text(strip=True) if tournament_element else 'Unknown Tournament'
            
            return match_info
            
        except Exception as e:
            logger.error(f"Error extracting match info for {match_id}: {e}")
            return None
    
    def _parse_map_data(self, soup: BeautifulSoup, match_info: Dict) -> List[Dict]:
        """Parse all map data from the match page"""
        maps_data = []
        
        try:
            # Find all map containers
            map_containers = soup.select('.vm-stats-game')
            
            for map_idx, map_container in enumerate(map_containers):
                map_data = {'map_number': map_idx + 1}
                
                # Extract map name
                map_name_elem = map_container.select_one('.map span')
                map_data['map_name'] = map_name_elem.get_text(strip=True) if map_name_elem else f'Map {map_idx + 1}'
                
                # Extract team names for this map
                team_names = []
                team_headers = map_container.select('.team-name')
                for team_header in team_headers[:2]:  # Only take first 2
                    team_names.append(team_header.get_text(strip=True))
                
                if len(team_names) < 2:
                    team_names = match_info.get('teams', ['Team A', 'Team B'])
                
                # Parse player stats for each team
                all_players = []
                team_tables = map_container.select('.wf-table-inset table')
                
                # Fallback if no tables found with primary selector
                if len(team_tables) == 0:
                    team_tables = map_container.select('table')
                
                for team_idx, table in enumerate(team_tables[:2]):  # Only process first 2 teams
                    team_name = team_names[team_idx] if team_idx < len(team_names) else f'Team {team_idx + 1}'
                    
                    player_rows = table.select('tbody tr')
                    
                    # Fallback if no tbody rows found
                    if len(player_rows) == 0:
                        player_rows = table.select('tr')
                        # Skip header row if present
                        if len(player_rows) > 0 and len(player_rows[0].find_all('th')) > 0:
                            player_rows = player_rows[1:]
                    
                    for row in player_rows:
                        player_stats = self._parse_player_stats(row, team_name)
                        if player_stats:
                            # Validate the player stats
                            is_valid, error_msg = player_stats.validate()
                            if is_valid:
                                all_players.append(player_stats)
                            else:
                                logger.warning(f"Invalid player stats for {player_stats.name}: {error_msg}")
                                self.stats['validation_failures'] += 1
                
                map_data['players'] = all_players
                
                # Only add map if we have reasonable player data
                if len(all_players) >= 8:  # At least 8 players (4v4 minimum)
                    maps_data.append(map_data)
                else:
                    logger.warning(f"Map {map_idx + 1} has only {len(all_players)} players - skipping")
                
        except Exception as e:
            logger.error(f"Error parsing map data: {e}")
            self.stats['parsing_errors'] += 1
        
        return maps_data
    
    def scrape_match(self, match_id: int) -> Optional[Dict]:
        """Scrape a single match with fixed parsing logic"""
        self.stats['matches_processed'] += 1
        
        try:
            # Get the page
            soup = self._safe_get_page(match_id)
            if not soup:
                self.stats['matches_failed'] += 1
                return None
            
            # Extract basic match info
            match_info = self._extract_match_info(soup, match_id)
            if not match_info:
                self.stats['matches_failed'] += 1
                return None
            
            # Parse map data
            maps_data = self._parse_map_data(soup, match_info)
            if not maps_data:
                self.stats['matches_failed'] += 1
                return None
            
            # Combine everything
            complete_match = {
                **match_info,
                'map_stats': maps_data,
                'scraped_at': datetime.now().isoformat()
            }
            
            # Final validation - check if data looks reasonable
            total_players = sum(len(map_data['players']) for map_data in maps_data)
            avg_kills_per_player = sum(
                sum(player.kills for player in map_data['players'])
                for map_data in maps_data
            ) / max(total_players, 1)
            
            if avg_kills_per_player > 50:  # Still too high
                logger.warning(f"Match {match_id} has suspicious avg kills: {avg_kills_per_player:.1f}")
                self.stats['validation_failures'] += 1
                return None
            
            self.stats['matches_successful'] += 1
            return complete_match
            
        except Exception as e:
            logger.error(f"Error scraping match {match_id}: {e}")
            self.stats['matches_failed'] += 1
            return None
    
    def scrape_match_list(self, match_ids: List[int], save_to_db: bool = True) -> Dict:
        """Scrape a list of matches"""
        print(f"🎮 Starting scraping of {len(match_ids)} matches...")
        self.stats['start_time'] = datetime.now()
        
        successful_matches = []
        
        for i, match_id in enumerate(match_ids):
            if i % 50 == 0:
                self._print_progress(i, len(match_ids))
            
            match_data = self.scrape_match(match_id)
            if match_data:
                successful_matches.append(match_data)
                
                if save_to_db:
                    # Convert to database format and save
                    # This would need to be implemented based on your database schema
                    pass
            
            # Progress update every 100 matches
            if i % 100 == 0 and i > 0:
                self._print_progress(i, len(match_ids))
        
        self._print_final_stats(successful_matches)
        return {
            'successful_matches': successful_matches,
            'stats': self.stats
        }
    
    def _print_progress(self, current: int, total: int):
        """Print scraping progress"""
        pct = (current / total) * 100
        success_rate = (self.stats['matches_successful'] / max(self.stats['matches_processed'], 1)) * 100
        
        print(f"📊 Progress: {current}/{total} ({pct:.1f}%) | "
              f"Success rate: {success_rate:.1f}% | "
              f"Validation failures: {self.stats['validation_failures']}")
    
    def _print_final_stats(self, successful_matches: List[Dict]):
        """Print final scraping statistics"""
        elapsed = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        print(f"\n🎉 SCRAPING COMPLETED!")
        print(f"⏱️  Total time: {elapsed}")
        print(f"📊 Matches processed: {self.stats['matches_processed']:,}")
        print(f"✅ Successful: {self.stats['matches_successful']:,}")
        print(f"❌ Failed: {self.stats['matches_failed']:,}")
        print(f"⚠️  Validation failures: {self.stats['validation_failures']:,}")
        print(f"🔧 Parsing errors: {self.stats['parsing_errors']:,}")
        
        if successful_matches:
            # Calculate data quality stats
            all_players = []
            for match in successful_matches:
                for map_data in match.get('map_stats', []):
                    all_players.extend(map_data.get('players', []))
            
            if all_players:
                avg_kills = sum(p.kills for p in all_players) / len(all_players)
                avg_deaths = sum(p.deaths for p in all_players) / len(all_players)
                avg_assists = sum(p.assists for p in all_players) / len(all_players)
                
                print(f"\n📈 DATA QUALITY (New vs Old):")
                print(f"   Average kills: {avg_kills:.1f} (vs 201.8 corrupted)")
                print(f"   Average deaths: {avg_deaths:.1f} (vs 0.0 corrupted)")
                print(f"   Average assists: {avg_assists:.1f} (vs 0.0 corrupted)")
                print(f"   Total validated players: {len(all_players):,}")

def quick_test(match_ids: List[int] = None):
    """Quick test of the fixed scraper"""
    if not match_ids:
        # Load some match IDs from our extracted list
        try:
            with open('match_ids_to_scrape.json', 'r') as f:
                data = json.load(f)
                match_ids = data['match_ids'][:5]  # Test first 5
        except FileNotFoundError:
            print("❌ No match IDs file found. Run extract_match_ids.py first")
            return
    
    print("🧪 TESTING FIXED VLR.GG SCRAPER")
    print("=" * 50)
    
    scraper = FixedVLRScraper()
    results = scraper.scrape_match_list(match_ids[:5], save_to_db=False)
    
    print("✅ Test completed!")
    return results

if __name__ == "__main__":
    quick_test() 