#!/usr/bin/env python3
"""
Enhanced Data Loader for Valorant Kill Line Prediction
Processes scraped match data to create training datasets
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import glob
from datetime import datetime
import sys

# Add the Scraper directory to path to import db_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
from db_utils import get_connection, get_players, get_teams, get_players_by_team

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
                map_name = map_stat.get('map_name', 'Unknown')
                flat_players = map_stat.get('flat_players', [])
                
                for player_data in flat_players:
                    try:
                        player = MatchPlayer(
                            name=player_data.get('name', ''),
                            team=player_data.get('team', ''),
                            kills=int(player_data.get('kills', 0)),
                            deaths=int(player_data.get('deaths', 0)),
                            assists=int(player_data.get('assists', 0)),
                            rating=float(player_data.get('rating', 0)),
                            acs=float(player_data.get('acs', 0)),
                            adr=float(player_data.get('adr', 0)),
                            kast=float(player_data.get('kast', 0)),
                            kd_ratio=float(player_data.get('kd_ratio', 0)),
                            headshot_percentage=float(player_data.get('headshot_percentage', 0)),
                            first_kills=int(player_data.get('first_kills', 0)),
                            first_deaths=int(player_data.get('first_deaths', 0)),
                            clutches=int(player_data.get('clutches', 0)),
                            map_name=map_name,
                            match_date=date,
                            tournament=tournament
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
                
                # Get player's database stats
                player_db_row = None
                if not player_db_stats.empty:
                    player_db_row = player_db_stats[
                        (player_db_stats['name'] == player.name) & 
                        (player_db_stats['team'] == player.team)
                    ]
                
                # Create feature vector
                features = {
                    'match_id': match.match_id,
                    'player_name': player.name,
                    'team': player.team,
                    'opponent_team': [t for t in match.teams if t != player.team][0] if len(match.teams) > 1 else 'Unknown',
                    'map_name': player.map_name,
                    'tournament': player.tournament,
                    'match_date': player.match_date,
                    
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
        
        # Create features for kill prediction
        df = self.create_kill_prediction_features(df)
        
        # Select features for training - ONLY historical/contextual features
        # Remove features that leak current match performance information
        feature_columns = [
            # Database stats (historical performance) - excluding db_kills_per_round since it's the target
            'db_rating', 'db_average_combat_score', 'db_kill_deaths',
            'db_assists_per_round', 'db_first_kills_per_round', 'db_first_deaths_per_round',
            'db_headshot_percentage', 'db_clutch_success_percentage',
            
            # Match context features
            'opponent_team_strength', 'team_strength', 'map_familiarity',
            'recent_form', 'tournament_importance'
        ]
        
        # Target: kills per round from database (not calculated from match totals)
        target_column = 'db_kills_per_round'
        
        # Remove rows with missing values
        df_clean = df[feature_columns + [target_column]].dropna()
        
        if df_clean.empty:
            print("No data remaining after cleaning.")
            return pd.DataFrame(), pd.Series()
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        print(f"Final training dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    
    def create_kill_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for predicting actual kills"""
        print("Creating kill prediction features...")
        
        # Calculate derived features
        df['total_rounds'] = df['match_kills'] + df['match_deaths'] + df['match_assists']
        df['kills_per_round'] = df['match_kills'] / df['total_rounds'].replace(0, 1)
        df['deaths_per_round'] = df['match_deaths'] / df['total_rounds'].replace(0, 1)
        df['assists_per_round'] = df['match_assists'] / df['total_rounds'].replace(0, 1)
        
        # Performance ratios (historical vs current)
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
        
        # Create opponent strength feature (simplified)
        df['opponent_team_strength'] = 1.0  # Placeholder - would need opponent data
        
        # Create team strength feature
        df['team_strength'] = df['team_avg_rating']
        
        # Create map familiarity feature (simplified)
        df['map_familiarity'] = 0.5  # Placeholder - would need player map history
        
        # Create recent form feature (simplified)
        df['recent_form'] = df['performance_ratio']
        
        # Create tournament importance feature (simplified)
        df['tournament_importance'] = 0.5  # Placeholder - would need tournament tier data
        
        return df 