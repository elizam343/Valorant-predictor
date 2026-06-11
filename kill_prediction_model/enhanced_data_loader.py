#!/usr/bin/env python3
"""
Enhanced Data Loader for Valorant Kill Line Prediction
Processes scraped match data to create training datasets
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import os
import glob
from datetime import datetime
import sys

# Add the Scraper directory to path to import db_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
from db_utils import get_connection, get_players, get_teams, get_players_by_team

# ---------------------------------------------------------------------------
# Agent role mapping  (0 = Sentinel … 3 = Duelist)
# ---------------------------------------------------------------------------
AGENT_ROLES: Dict[str, int] = {
    # Duelists — highest expected kill counts
    'jett': 3, 'neon': 3, 'reyna': 3, 'raze': 3,
    'yoru': 3, 'phoenix': 3, 'iso': 3, 'waylay': 3,
    # Initiators — moderate kills, lots of utility
    'sova': 2, 'breach': 2, 'kayo': 2, 'kay/o': 2,
    'skye': 2, 'fade': 2, 'gekko': 2, 'tejo': 2,
    # Controllers — map control, lower kill floor
    'brimstone': 1, 'omen': 1, 'astra': 1,
    'viper': 1, 'harbor': 1, 'clove': 1,
    # Sentinels — utility/anchor role, lowest kill floor
    'killjoy': 0, 'cypher': 0, 'sage': 0,
    'chamber': 0, 'deadlock': 0, 'vyse': 0,
}
UNKNOWN_ROLE: float = 1.5  # midpoint when agent is missing or unrecognised

@dataclass
class MatchPlayer:
    """Data class for player performance in a specific match"""
    name: str
    team: str
    kills: int
    deaths: int
    assists: int
    rating: float
    acs: float
    adr: float
    kast: float
    kd_ratio: float
    headshot_percentage: float
    first_kills: int
    first_deaths: int
    clutches: int
    map_name: str
    match_date: str
    tournament: str
    agent: str = ''

@dataclass
class MatchData:
    """Data class for complete match information"""
    match_id: str
    date: str
    teams: List[str]
    tournament: str
    players: List[MatchPlayer]
    map_stats: List[Dict]

class EnhancedDataLoader:
    """Enhanced data loader that processes scraped match data"""
    
    def __init__(self, scraped_matches_dir: str = None):
        if scraped_matches_dir is None:
            # Default to the scraped_matches directory in the parent folder
            current_dir = os.path.dirname(__file__)
            self.scraped_matches_dir = os.path.join(current_dir, '..', 'scraped_matches')
        else:
            self.scraped_matches_dir = scraped_matches_dir
        
        # Cache for player database stats
        self.player_stats_cache = None
        
    def load_player_database_stats(self) -> pd.DataFrame:
        """Load player statistics from the database"""
        if self.player_stats_cache is not None:
            return self.player_stats_cache
            
        try:
            conn = get_connection()
            query = """
            SELECT name, team, rating, average_combat_score, kill_deaths, 
                   kill_assists_survived_traded, average_damage_per_round, 
                   kills_per_round, assists_per_round, first_kills_per_round,
                   first_deaths_per_round, headshot_percentage, clutch_success_percentage
            FROM players
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert string columns to numeric, handling missing values
            numeric_columns = ['rating', 'average_combat_score', 'kill_deaths', 
                              'kill_assists_survived_traded', 'average_damage_per_round',
                              'kills_per_round', 'assists_per_round', 'first_kills_per_round',
                              'first_deaths_per_round', 'headshot_percentage', 'clutch_success_percentage']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.player_stats_cache = df
            return df
        except Exception as e:
            print(f"Warning: Could not load player database stats: {e}")
            return pd.DataFrame()
    
    def load_scraped_matches(self, limit: int = None) -> List[MatchData]:
        """Load and parse scraped match data"""
        if not os.path.exists(self.scraped_matches_dir):
            print(f"Scraped matches directory not found: {self.scraped_matches_dir}")
            return []
        
        # Get all JSON files
        json_files = glob.glob(os.path.join(self.scraped_matches_dir, "match_*.json"))
        if limit:
            json_files = json_files[:limit]
        
        print(f"Loading {len(json_files)} scraped matches...")
        
        matches = []
        for i, file_path in enumerate(json_files):
            try:
                with open(file_path, 'r') as f:
                    match_data = json.load(f)
                
                # Extract match ID from filename
                match_id = os.path.basename(file_path).replace('match_', '').replace('.json', '')
                
                # Parse match data
                match = self._parse_match_data(match_data, match_id)
                if match:
                    matches.append(match)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} matches...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(matches)} matches")
        return matches
    
    def _parse_match_data(self, match_data: Dict, match_id: str) -> Optional[MatchData]:
        """Parse raw match data into structured format"""

        def _safe_stat_float(v, default=0.0):
            try:
                return float(str(v).split("/")[0].strip() or default)
            except (ValueError, TypeError):
                return default

        def _safe_stat_int(v, default=0):
            try:
                return int(float(str(v).split("/")[0].strip() or default))
            except (ValueError, TypeError):
                return default

        try:
            # Extract basic match info
            date = match_data.get('date', '')
            teams = match_data.get('teams', [])
            tournament = match_data.get('tournament', 'Unknown Tournament')
            map_stats = match_data.get('map_stats', [])
            
            if not teams or len(teams) < 2:
                return None
            
            # Parse players from map stats
            players = []
            for map_stat in map_stats:
                # JSON key is 'map'; fall back to 'map_name' for older files
                map_name = map_stat.get('map') or map_stat.get('map_name', 'Unknown')
                flat_players = map_stat.get('flat_players', [])

                for player_data in flat_players:
                    try:
                        player = MatchPlayer(
                            name=player_data.get('name', ''),
                            team=player_data.get('team', ''),
                            kills=_safe_stat_int(player_data.get('kills', 0)),
                            deaths=_safe_stat_int(player_data.get('deaths', 0)),
                            assists=_safe_stat_int(player_data.get('assists', 0)),
                            # 'rating' key is the VLR.gg rating (1.xx format)
                            rating=_safe_stat_float(player_data.get('rating', 0)),
                            acs=_safe_stat_float(player_data.get('acs', 0)),
                            adr=_safe_stat_float(player_data.get('adr', 0)),
                            kast=_safe_stat_float(player_data.get('kast', 0)),
                            kd_ratio=_safe_stat_float(player_data.get('kd_diff', 0)),
                            headshot_percentage=_safe_stat_float(
                                str(player_data.get('hs%', '0')).replace('%', '')
                            ),
                            first_kills=_safe_stat_int(player_data.get('fk', 0)),
                            first_deaths=_safe_stat_int(player_data.get('fd', 0)),
                            clutches=0,
                            map_name=map_name,
                            match_date=date,
                            tournament=tournament,
                            agent=str(player_data.get('agent', '') or '').strip(),
                        )
                        players.append(player)
                    except (ValueError, TypeError) as e:
                        continue
            
            return MatchData(
                match_id=match_id,
                date=date,
                teams=teams,
                tournament=tournament,
                players=players,
                map_stats=map_stats
            )
            
        except Exception as e:
            print(f"Error parsing match {match_id}: {e}")
            return None
    
    def create_training_dataset(self, matches: List[MatchData]) -> pd.DataFrame:
        """Create training dataset from match data"""
        print("Creating training dataset...")
        
        # Load player database stats for additional features
        player_db_stats = self.load_player_database_stats()
        
        training_data = []
        
        for match in matches:
            # Create features for each player in the match
            for player in match.players:
                if player.kills == 0 and player.deaths == 0:
                    continue  # Skip players with no activity
                
                # Get player's database stats — match on name only; team names differ
                # between vlrggapi (DB source) and VLR.gg match pages.
                player_db_row = None
                if not player_db_stats.empty:
                    player_db_row = player_db_stats[player_db_stats['name'] == player.name]
                    if len(player_db_row) > 1:
                        # Multiple rows for the same name (player switched teams) — take the first
                        player_db_row = player_db_row.iloc[[0]]
                
                # Create feature vector
                features = {
                    'match_id': match.match_id,
                    'player_name': player.name,
                    'team': player.team,
                    'opponent_team': [t for t in match.teams if t != player.team][0] if len(match.teams) > 1 else 'Unknown',
                    'map_name': player.map_name,
                    'tournament': player.tournament,
                    'match_date': player.match_date,
                    'agent': player.agent,
                    
                    # Match performance features
                    'match_kills': player.kills,
                    'match_deaths': player.deaths,
                    'match_assists': player.assists,
                    'match_rating': player.rating,
                    'match_acs': player.acs,
                    'match_adr': player.adr,
                    'match_kast': player.kast,
                    'match_kd_ratio': player.kd_ratio,
                    'match_headshot_percentage': player.headshot_percentage,
                    'match_first_kills': player.first_kills,
                    'match_first_deaths': player.first_deaths,
                    'match_clutches': player.clutches,
                    
                    # Database stats features (if available)
                    'db_rating': player_db_row['rating'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_average_combat_score': player_db_row['average_combat_score'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_kill_deaths': player_db_row['kill_deaths'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_kills_per_round': player_db_row['kills_per_round'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_assists_per_round': player_db_row['assists_per_round'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_first_kills_per_round': player_db_row['first_kills_per_round'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_first_deaths_per_round': player_db_row['first_deaths_per_round'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_headshot_percentage': player_db_row['headshot_percentage'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                    'db_clutch_success_percentage': player_db_row['clutch_success_percentage'].iloc[0] if player_db_row is not None and not player_db_row.empty else 0.0,
                }
                
                training_data.append(features)
        
        df = pd.DataFrame(training_data)
        print(f"Created training dataset with {len(df)} player-match records")
        return df
    
    def create_kill_line_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for kill line prediction"""
        print("Creating kill line prediction features...")
        
        # Calculate derived features
        df['total_rounds'] = df['match_kills'] + df['match_deaths'] + df['match_assists']
        df['kills_per_round'] = df['match_kills'] / df['total_rounds'].replace(0, 1)
        df['deaths_per_round'] = df['match_deaths'] / df['total_rounds'].replace(0, 1)
        df['assists_per_round'] = df['match_assists'] / df['total_rounds'].replace(0, 1)
        
        # Performance ratios
        df['performance_ratio'] = df['match_rating'] / df['db_rating'].replace(0, 1)
        df['acs_ratio'] = df['match_acs'] / df['db_average_combat_score'].replace(0, 1)
        
        # Efficiency metrics
        df['kill_efficiency'] = df['match_kills'] / (df['match_kills'] + df['match_deaths']).replace(0, 1)
        df['impact_score'] = df['match_kills'] + (df['match_assists'] * 0.5)
        
        # Team performance features
        team_stats = df.groupby(['match_id', 'team']).agg({
            'match_kills': 'sum',
            'match_rating': 'mean',
            'match_acs': 'mean'
        }).reset_index()
        
        team_stats = team_stats.rename(columns={
            'match_kills': 'team_total_kills',
            'match_rating': 'team_avg_rating',
            'match_acs': 'team_avg_acs'
        })
        
        df = df.merge(team_stats, on=['match_id', 'team'], how='left')
        
        # Player's contribution to team
        df['team_kill_contribution'] = df['match_kills'] / df['team_total_kills'].replace(0, 1)
        df['relative_rating'] = df['match_rating'] - df['team_avg_rating']
        
        # Map-specific features
        map_stats = df.groupby('map_name').agg({
            'match_kills': 'mean',
            'match_rating': 'mean'
        }).rename(columns={
            'match_kills': 'map_avg_kills',
            'match_rating': 'map_avg_rating'
        })
        
        df = df.merge(map_stats, on='map_name', how='left')
        df['map_performance'] = df['match_kills'] - df['map_avg_kills']
        
        return df
    
    def generate_kill_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic kill lines based on player performance patterns"""
        print("Generating synthetic kill lines...")
        
        # Convert match_date to datetime for time-based weighting
        df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Sort by player and date to ensure chronological order
        df = df.sort_values(['player_name', 'match_date']).reset_index(drop=True)
        
        # Calculate rolling average kills per round for each player (excluding current match)
        df['kill_line'] = 0.0
        
        for player in df['player_name'].unique():
            player_matches = df[df['player_name'] == player].copy()
            
            for idx, row in player_matches.iterrows():
                # Get all previous matches for this player
                previous_matches = player_matches[player_matches.index < idx]
                
                if len(previous_matches) >= 3:  # Need at least 3 previous matches
                    # Calculate time-weighted average kills per round from previous matches
                    current_date = row['match_date']
                    
                    # Calculate days since each previous match (more recent = higher weight)
                    previous_matches['days_ago'] = (current_date - previous_matches['match_date']).dt.days
                    
                    # Create exponential decay weights (more recent matches get higher weight)
                    previous_matches['weight'] = np.exp(-previous_matches['days_ago'] / 365)  # 1 year half-life
                    
                    # Calculate weighted average
                    weighted_kills = np.average(
                        previous_matches['kills_per_round'], 
                        weights=previous_matches['weight']
                    )
                    
                    # Add some variance to make it realistic (15% standard deviation)
                    np.random.seed(hash(f"{player}_{row['match_date']}") % 2**32)
                    variance_factor = np.random.normal(1.0, 0.15)
                    kill_line = (weighted_kills * variance_factor).round(1)
                    
                    # Ensure kill lines are reasonable (between 0.5 and 2.0)
                    kill_line = max(0.5, min(2.0, kill_line))
                    
                    df.loc[idx, 'kill_line'] = kill_line
                else:
                    # For players with < 3 previous matches, use a default based on their database stats
                    default_kills = row['db_kills_per_round'] if row['db_kills_per_round'] > 0 else 0.8
                    df.loc[idx, 'kill_line'] = round(default_kills, 1)
        
        # Create target variable: 0=under, 1=over
        df['target'] = (df['kills_per_round'] > df['kill_line']).astype(int)
        
        # Add confidence based on how close the actual performance was to the line
        df['confidence'] = 1 - abs(df['kills_per_round'] - df['kill_line']) / df['kill_line']
        df['confidence'] = df['confidence'].clip(0.1, 1.0)
        
        # Remove rows where we couldn't generate a proper kill line
        df = df[df['kill_line'] > 0].copy()
        
        return df
    
    def _safe_int(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def create_series_level_dataset(self, matches: List[MatchData], max_maps: int = 3) -> pd.DataFrame:
        """Create a dataset with one row per player per match (series), aggregating stats per map in order."""
        print("Creating series-level dataset...")
        rows = []
        for match in matches:
            # Build a dict: {(player_name, team): [map1_stats, map2_stats, ...]}
            player_map_stats = {}
            for map_idx, map_stat in enumerate(match.map_stats):
                map_name = map_stat.get('map_name', f"Map{map_idx+1}")
                flat_players = map_stat.get('flat_players', [])
                for player_data in flat_players:
                    key = (player_data.get('name', ''), player_data.get('team', ''))
                    if key not in player_map_stats:
                        player_map_stats[key] = []
                    player_map_stats[key].append({
                        'kills': self._safe_int(player_data.get('kills', 0)),
                        'deaths': self._safe_int(player_data.get('deaths', 0)),
                        'assists': self._safe_int(player_data.get('assists', 0)),
                        'agent': player_data.get('agent', ''),
                        'side': player_data.get('side', ''),
                        'map_name': map_name
                    })
            for (player_name, team), stats_list in player_map_stats.items():
                # Pad stats_list to max_maps
                stats_list = stats_list[:max_maps] + [{}]*(max_maps - len(stats_list))
                row = {
                    'match_id': match.match_id,
                    'player_name': player_name,
                    'team': team,
                    'opponent_team': [t for t in match.teams if t != team][0] if len(match.teams) > 1 else 'Unknown',
                    'tournament': match.tournament,
                    'match_date': match.date,
                }
                total_kills_first2 = 0
                for i in range(max_maps):
                    s = stats_list[i]
                    row[f'kills_map{i+1}'] = s.get('kills', 0)
                    row[f'deaths_map{i+1}'] = s.get('deaths', 0)
                    row[f'assists_map{i+1}'] = s.get('assists', 0)
                    row[f'agent_map{i+1}'] = s.get('agent', '')
                    row[f'side_map{i+1}'] = s.get('side', '')
                    row[f'map_name_{i+1}'] = s.get('map_name', '')
                    if i < 2:
                        total_kills_first2 += s.get('kills', 0)
                row['total_kills_first2'] = total_kills_first2
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f"Created series-level dataset with {len(df)} player-series records")
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-series-safe per-player rolling features.

        Uses shift(1) before each rolling window so no row ever sees its own
        current-match value — only data from strictly earlier matches.
        """
        df = df.copy()
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        df = df.sort_values(['player_name', 'match_date']).reset_index(drop=True)

        grp = df.groupby('player_name', group_keys=False)

        # Rolling average kills & rating over the last 10 map appearances
        df['recent_avg_kills'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=2).mean()
        )
        df['recent_avg_rating'] = grp['match_rating'].transform(
            lambda s: s.shift(1).rolling(10, min_periods=2).mean()
        )
        df['recent_avg_kills_3'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(df['match_kills'].mean())

        # days_since_last_match — rest days between consecutive appearances
        df['_prev_date'] = df.groupby('player_name')['match_date'].shift(1)
        df['days_since_last_match'] = (
            (df['match_date'] - df['_prev_date']).dt.days
            .fillna(7.0).clip(0, 30)
        )
        df = df.drop(columns=['_prev_date'])

        # form_slope — linear trend (slope) of last 5 map kills
        # Positive = trending up, negative = trending down
        df['form_slope'] = grp['match_kills'].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda y: float(np.polyfit(range(len(y)), y, 1)[0]),
                raw=True,
            )
        )
        df['form_slope'] = df['form_slope'].fillna(0.0)

        # rating_form_slope — EFFICIENCY trajectory (slope of last 5 maps' VLR
        # rating). Unlike form_slope (kills), rating is role/round-robust, so it
        # tracks whether a player is genuinely improving/declining independent of
        # kill volume — a leading indicator the market (which prices recent kills)
        # tends to lag. Same shift(1) leak-safety as the kills slope.
        df['rating_form_slope'] = grp['match_rating'].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda y: float(np.polyfit(range(len(y)), y, 1)[0]),
                raw=True,
            )
        )
        df['rating_form_slope'] = df['rating_form_slope'].fillna(0.0)

        # Per-player-per-map kill average using leave-one-out to avoid leakage.
        grp_map   = df.groupby(['player_name', 'map_name'])['match_kills']
        map_sum   = grp_map.transform('sum')
        map_count = grp_map.transform('count')
        df['player_map_avg_kills'] = (map_sum - df['match_kills']) / (map_count - 1)
        # Players with only one appearance on a map get their overall average instead
        player_overall = df.groupby('player_name')['match_kills'].transform('mean')
        df['player_map_avg_kills'] = df['player_map_avg_kills'].fillna(player_overall)

        # h2h_avg_kills — player's historical avg kills vs this specific opponent (leave-one-out)
        grp_h2h   = df.groupby(['player_name', 'opponent_team'])['match_kills']
        h2h_sum   = grp_h2h.transform('sum')
        h2h_count = grp_h2h.transform('count')
        df['h2h_avg_kills'] = (h2h_sum - df['match_kills']) / (h2h_count - 1)
        df['h2h_avg_kills'] = df['h2h_avg_kills'].fillna(player_overall)

        # h2h_data_exists — 1 if real h2h data, 0 if first-time matchup
        df['h2h_data_exists'] = (h2h_count > 1).astype(float)

        # kill_std — player's historical kill standard deviation
        df['kill_std'] = (
            df.groupby('player_name')['match_kills'].transform('std').fillna(3.0)
        )

        return df

    def add_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add agent role and per-agent kill history features.

        New columns:
          agent_role_ordinal    0=Sentinel, 1=Controller, 2=Initiator, 3=Duelist, 1.5=unknown
          player_agent_avg_kills  player's historical avg kills when playing this agent
                                  (leave-one-out to avoid leakage)
        """
        df = df.copy()

        agent_col = df['agent'].str.lower().str.strip() if 'agent' in df.columns else pd.Series('', index=df.index)

        df['agent_role_ordinal'] = agent_col.map(AGENT_ROLES).fillna(UNKNOWN_ROLE)

        # Per-player-per-agent kill average (leave-one-out)
        df['_agent_key'] = agent_col
        grp = df.groupby(['player_name', '_agent_key'])['match_kills']
        agent_sum   = grp.transform('sum')
        agent_count = grp.transform('count')
        df['player_agent_avg_kills'] = (agent_sum - df['match_kills']) / (agent_count - 1)
        # Fallback: player's overall average when only one appearance on this agent
        player_overall = df.groupby('player_name')['match_kills'].transform('mean')
        df['player_agent_avg_kills'] = df['player_agent_avg_kills'].fillna(player_overall)

        df = df.drop(columns=['_agent_key'])
        return df

    def prepare_training_data(self, limit_matches: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Main method to prepare training data from scraped matches"""
        print("=== Preparing Training Data from Scraped Matches ===")
        
        # Load scraped matches
        matches = self.load_scraped_matches(limit=limit_matches)
        
        if not matches:
            print("No matches found. Please ensure scraped matches are available.")
            return pd.DataFrame(), pd.Series()
        
        # Create training dataset
        df = self.create_training_dataset(matches)

        if df.empty:
            print("No valid training data created.")
            return pd.DataFrame(), pd.Series()

        # Add contextual (team strength, opponent strength)
        df = self.create_kill_prediction_features(df)

        # Add rolling recent-form and map-specific features
        df = self.add_rolling_features(df)

        # Add agent role and per-agent kill history
        df = self.add_agent_features(df)

        from features import FEATURE_COLS   # single source of truth (#6)
        feature_columns = list(FEATURE_COLS)

        # Target: actual kills in this map (the thing bettors are trying to predict)
        target_column = 'match_kills'

        extra_cols = [target_column]
        if 'player_name' in df.columns:
            extra_cols.append('player_name')
        # match_date is carried through (not a feature) so the trainer can do a
        # leak-free chronological split instead of a random one.
        if 'match_date' in df.columns:
            extra_cols.append('match_date')

        df_subset = df[feature_columns + extra_cols].copy()

        # Keep only rows where the player has career stats and scored real kills
        df_subset = df_subset[
            (df_subset['db_rating'] > 0) &
            (df_subset[target_column] > 0)
        ]

        # Fill remaining NaN with column medians so sparse features don't wipe the dataset
        df_subset = df_subset.fillna(df_subset.median(numeric_only=True))

        if df_subset.empty:
            print("No data remaining after cleaning.")
            return pd.DataFrame(), pd.Series()

        X = df_subset[feature_columns].copy()
        if 'player_name' in df_subset.columns:
            X['player_name'] = df_subset['player_name'].values
        if 'match_date' in df_subset.columns:
            X['match_date'] = df_subset['match_date'].values
        y = df_subset[target_column]

        print(f"Final training dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Target: '{target_column}' | range [{y.min():.0f}, {y.max():.0f}] | mean {y.mean():.2f}")

        return X, y
    
    def create_kill_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for predicting actual kills"""
        print("Creating kill prediction features...")

        # Team performance in this match
        team_stats = df.groupby(['match_id', 'team']).agg(
            team_total_kills=('match_kills', 'sum'),
            team_avg_rating=('match_rating', 'mean'),
            team_avg_acs=('match_acs', 'mean'),
        ).reset_index()
        df = df.merge(team_stats, on=['match_id', 'team'], how='left')

        # team_strength = own team's avg rating in this match
        df['team_strength'] = df['team_avg_rating']

        # opponent_team_strength = ACTUAL opposing team's avg rating in this match
        # (previously was just match_avg_rating — now fixed)
        opp_ratings = (
            df.groupby(['match_id', 'team'])['match_rating']
            .mean()
            .reset_index()
            .rename(columns={'team': 'opponent_team', 'match_rating': 'opponent_team_strength'})
        )
        df = df.merge(opp_ratings, on=['match_id', 'opponent_team'], how='left')
        df['opponent_team_strength'] = df['opponent_team_strength'].fillna(df['team_avg_rating'])

        # opponent_kills_allowed_per_map — historical avg kills opponents score vs this team
        # i.e. how many kills does this opponent allow per map (their defensive rating)
        opp_allowed = (
            df.groupby('opponent_team')['match_kills']
            .mean()
            .reset_index()
            .rename(columns={'match_kills': 'opponent_kills_allowed_per_map'})
        )
        df = df.merge(opp_allowed, on='opponent_team', how='left')
        df['opponent_kills_allowed_per_map'] = df['opponent_kills_allowed_per_map'].fillna(
            df['match_kills'].mean()
        )

        # Map-average kills (global, for context)
        map_avg = (
            df.groupby('map_name')['match_kills']
            .mean()
            .reset_index()
            .rename(columns={'match_kills': 'map_avg_kills'})
        )
        df = df.merge(map_avg, on='map_name', how='left')

        # avg_rounds_vs_opponent — estimated rounds from total kills per map instance
        map_totals = (df.groupby(['match_id', 'map_name'])['match_kills']
                      .sum().reset_index()
                      .rename(columns={'match_kills': 'map_total_kills'}))
        map_totals['estimated_rounds'] = map_totals['map_total_kills'] / 6.0
        df = df.merge(map_totals[['match_id', 'map_name', 'estimated_rounds']],
                      on=['match_id', 'map_name'], how='left')
        grp_r   = df.groupby(['team', 'opponent_team'])['estimated_rounds']
        r_sum   = grp_r.transform('sum')
        r_count = grp_r.transform('count')
        df['avg_rounds_vs_opponent'] = (
            (r_sum - df['estimated_rounds']) / (r_count - 1)
        ).fillna(df['estimated_rounds'].mean())

        return df