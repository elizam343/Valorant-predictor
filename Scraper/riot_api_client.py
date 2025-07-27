"""
Riot Games API Client for Valorant Match Data
Clean, official data with proper validation
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerMatchStats:
    """Clean data structure for player match statistics"""
    player_puuid: str
    player_name: str
    team_id: str
    agent: str
    kills: int
    deaths: int
    assists: int
    score: int
    damage: int
    headshots: int
    first_kills: int
    first_deaths: int
    rounds_played: int
    
    def validate(self) -> Tuple[bool, str]:
        """Validate that stats are reasonable for Valorant"""
        if not (0 <= self.kills <= 50):
            return False, f"Invalid kills: {self.kills}"
        if not (0 <= self.deaths <= 50):
            return False, f"Invalid deaths: {self.deaths}"
        if not (0 <= self.assists <= 50):
            return False, f"Invalid assists: {self.assists}"
        if self.score < 0:
            return False, f"Invalid score: {self.score}"
        if self.damage < 0:
            return False, f"Invalid damage: {self.damage}"
        if not (0 <= self.headshots <= self.kills):
            return False, f"Invalid headshots: {self.headshots} > kills: {self.kills}"
        if not (0 <= self.first_kills <= 10):
            return False, f"Invalid first kills: {self.first_kills}"
        if not (0 <= self.first_deaths <= 10):
            return False, f"Invalid first deaths: {self.first_deaths}"
        if not (5 <= self.rounds_played <= 30):
            return False, f"Invalid rounds played: {self.rounds_played}"
            
        return True, "Valid"

@dataclass
class MatchData:
    """Clean data structure for match information"""
    match_id: str
    game_start: datetime
    game_length: int
    map_name: str
    game_mode: str
    is_ranked: bool
    season_id: str
    rounds_played: int
    team_stats: List[Dict]
    player_stats: List[PlayerMatchStats]
    
    def validate(self) -> Tuple[bool, str]:
        """Validate match data"""
        if not self.match_id:
            return False, "Missing match ID"
        if not self.map_name:
            return False, "Missing map name"
        if not (5 <= self.rounds_played <= 30):
            return False, f"Invalid rounds played: {self.rounds_played}"
        if len(self.player_stats) != 10:
            return False, f"Expected 10 players, got {len(self.player_stats)}"
            
        # Validate all player stats
        for player_stat in self.player_stats:
            is_valid, error = player_stat.validate()
            if not is_valid:
                return False, f"Player {player_stat.player_name}: {error}"
                
        return True, "Valid"

class RiotAPIClient:
    """Official Riot Games API client for Valorant data"""
    
    def __init__(self, api_key: str, region: str = "na"):
        self.api_key = api_key
        self.region = region.lower()
        self.base_url = self._get_base_url()
        self.session = requests.Session()
        self.session.headers.update({
            'X-Riot-Token': api_key,
            'Accept': 'application/json'
        })
        self.rate_limit_delay = 1.2  # seconds between requests
        self.last_request_time = 0
        
    def _get_base_url(self) -> str:
        """Get the correct API base URL for the region"""
        region_clusters = {
            'na': 'americas',
            'br': 'americas', 
            'latam': 'americas',
            'eu': 'europe',
            'ap': 'asia',
            'kr': 'asia'
        }
        cluster = region_clusters.get(self.region, 'americas')
        return f"https://{cluster}.api.riotgames.com"
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited API request"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)
            elif response.status_code == 404:
                logger.info(f"Match not found: {endpoint}")
                return None
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_player_by_name(self, game_name: str, tag_line: str) -> Optional[Dict]:
        """Get player PUUID by Riot ID"""
        endpoint = f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._make_request(endpoint)
    
    def get_match_history(self, puuid: str, count: int = 20) -> Optional[List[str]]:
        """Get recent match IDs for a player"""
        endpoint = f"/val/match/v1/matchlists/by-puuid/{puuid}"
        params = {'size': min(count, 20)}  # API limit is 20
        
        response = self._make_request(endpoint, params)
        if response:
            return response.get('history', [])
        return None
    
    def get_match_details(self, match_id: str) -> Optional[MatchData]:
        """Get detailed match information"""
        endpoint = f"/val/match/v1/matches/{match_id}"
        response = self._make_request(endpoint)
        
        if not response:
            return None
            
        try:
            return self._parse_match_data(response)
        except Exception as e:
            logger.error(f"Error parsing match {match_id}: {e}")
            return None
    
    def _parse_match_data(self, raw_data: Dict) -> MatchData:
        """Parse raw API response into clean MatchData"""
        match_info = raw_data['matchInfo']
        
        # Basic match info
        match_id = match_info['matchId']
        game_start = datetime.fromtimestamp(match_info['gameStartMillis'] / 1000)
        game_length = match_info['gameLengthMillis'] // 1000  # Convert to seconds
        map_name = match_info['mapId'].split('/')[-1]  # Extract map name
        game_mode = match_info['queueId']
        is_ranked = game_mode == 'competitive'
        season_id = match_info.get('seasonId', '')
        
        # Round info
        rounds_played = len(raw_data['roundResults'])
        
        # Team stats
        team_stats = []
        for team in raw_data['teams']:
            team_stats.append({
                'team_id': team['teamId'],
                'won': team['won'],
                'rounds_won': team['roundsWon'],
                'rounds_lost': team['roundsLost']
            })
        
        # Player stats
        player_stats = []
        for player in raw_data['players']:
            # Get player stats
            stats = player['stats']
            
            player_stat = PlayerMatchStats(
                player_puuid=player['puuid'],
                player_name=player.get('gameName', 'Unknown'),
                team_id=player['teamId'],
                agent=player['characterId'].split('/')[-1],
                kills=stats['kills'],
                deaths=stats['deaths'],
                assists=stats['assists'],
                score=stats['score'],
                damage=stats.get('totalDamageDealtToEnemies', 0),
                headshots=stats.get('headshots', 0),
                first_kills=stats.get('firstKills', 0),
                first_deaths=stats.get('firstDeaths', 0),
                rounds_played=rounds_played
            )
            
            player_stats.append(player_stat)
        
        match_data = MatchData(
            match_id=match_id,
            game_start=game_start,
            game_length=game_length,
            map_name=map_name,
            game_mode=game_mode,
            is_ranked=is_ranked,
            season_id=season_id,
            rounds_played=rounds_played,
            team_stats=team_stats,
            player_stats=player_stats
        )
        
        return match_data
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        endpoint = "/val/content/v1/contents"
        response = self._make_request(endpoint)
        
        if response:
            logger.info("✅ Riot API connection successful!")
            return True
        else:
            logger.error("❌ Riot API connection failed!")
            return False
    
    def get_featured_matches(self, count: int = 10) -> List[str]:
        """Get recent featured match IDs (for initial data collection)"""
        # This is a simplified approach - in practice you'd need player PUUIDs
        # to get their match histories
        logger.info("Note: Riot API requires player PUUIDs to get matches.")
        logger.info("Consider using a seed list of known players to start.")
        return []

# Data validation functions
def validate_kill_stats(kills: int, deaths: int, assists: int, rounds: int) -> bool:
    """Validate kill statistics are reasonable"""
    if not (0 <= kills <= rounds * 2):  # Max ~2 kills per round
        return False
    if not (0 <= deaths <= rounds):  # Max 1 death per round
        return False
    if not (0 <= assists <= rounds * 3):  # Max ~3 assists per round
        return False
    return True

def validate_match_duration(duration_seconds: int) -> bool:
    """Validate match duration is reasonable"""
    min_duration = 10 * 60  # 10 minutes (very short match)
    max_duration = 90 * 60  # 90 minutes (very long match)
    return min_duration <= duration_seconds <= max_duration

# Example usage
if __name__ == "__main__":
    # Example usage (user will provide their API key)
    api_key = "YOUR_RIOT_API_KEY_HERE"
    client = RiotAPIClient(api_key, region="na")
    
    # Test connection
    if client.test_connection():
        print("🚀 Ready to collect clean Valorant data!")
    else:
        print("❌ Please check your API key and try again.") 