import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add the Scraper directory to path to import db_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
from db_utils import get_connection, get_players, get_teams, get_players_by_team

@dataclass
class PlayerStats:
    """Data class to hold player statistics"""
    name: str
    team: str
    rating: float
    average_combat_score: float
    kill_deaths: float
    kill_assists_survived_traded: float
    average_damage_per_round: float
    kills_per_round: float
    assists_per_round: float
    first_kills_per_round: float
    first_deaths_per_round: float
    headshot_percentage: float
    clutch_success_percentage: float

class DataLoader:
    """Handles loading and preprocessing player data from the database"""
    
    def __init__(self, db_path: str = "../Scraper/vlr_players.db"):
        self.db_path = db_path
        
    def load_all_players(self) -> pd.DataFrame:
        """Load all players from database into a pandas DataFrame"""
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
            
        return df
    
    def get_player_stats(self, player_name: str, team: str = None) -> Optional[PlayerStats]:
        """Get stats for a specific player"""
        conn = get_connection()
        if team:
            query = "SELECT * FROM players WHERE name = ? AND team = ?"
            params = (player_name, team)
        else:
            query = "SELECT * FROM players WHERE name = ?"
            params = (player_name,)
            
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PlayerStats(
                name=row[1],
                team=row[2],
                rating=float(row[3]) if row[3] else 0.0,
                average_combat_score=float(row[4]) if row[4] else 0.0,
                kill_deaths=float(row[5]) if row[5] else 0.0,
                kill_assists_survived_traded=float(row[6]) if row[6] else 0.0,
                average_damage_per_round=float(row[7]) if row[7] else 0.0,
                kills_per_round=float(row[8]) if row[8] else 0.0,
                assists_per_round=float(row[9]) if row[9] else 0.0,
                first_kills_per_round=float(row[10]) if row[10] else 0.0,
                first_deaths_per_round=float(row[11]) if row[11] else 0.0,
                headshot_percentage=float(row[12]) if row[12] else 0.0,
                clutch_success_percentage=float(row[13]) if row[13] else 0.0
            )
        return None
    
    def get_team_players(self, team: str) -> List[PlayerStats]:
        """Get all players from a specific team"""
        players_data = get_players_by_team(team)
        players = []
        
        for row in players_data:
            players.append(PlayerStats(
                name=row[1],
                team=row[2],
                rating=float(row[3]) if row[3] else 0.0,
                average_combat_score=float(row[4]) if row[4] else 0.0,
                kill_deaths=float(row[5]) if row[5] else 0.0,
                kill_assists_survived_traded=float(row[6]) if row[6] else 0.0,
                average_damage_per_round=float(row[7]) if row[7] else 0.0,
                kills_per_round=float(row[8]) if row[8] else 0.0,
                assists_per_round=float(row[9]) if row[9] else 0.0,
                first_kills_per_round=float(row[10]) if row[10] else 0.0,
                first_deaths_per_round=float(row[11]) if row[11] else 0.0,
                headshot_percentage=float(row[12]) if row[12] else 0.0,
                clutch_success_percentage=float(row[13]) if row[13] else 0.0
            ))
        
        return players
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the prediction model"""
        # Create derived features
        df['total_impact'] = df['kills_per_round'] + df['assists_per_round']
        df['survival_rate'] = 1 - df['first_deaths_per_round']
        df['efficiency'] = df['kills_per_round'] / (df['kills_per_round'] + df['deaths_per_round']).replace(0, 1)
        
        # Create team-level features
        team_stats = df.groupby('team').agg({
            'rating': 'mean',
            'kills_per_round': 'mean',
            'average_combat_score': 'mean'
        }).rename(columns={
            'rating': 'team_avg_rating',
            'kills_per_round': 'team_avg_kills',
            'average_combat_score': 'team_avg_acs'
        })
        
        df = df.merge(team_stats, on='team', how='left')
        
        # Create player's relative performance within team
        df['relative_rating'] = df['rating'] - df['team_avg_rating']
        df['relative_kills'] = df['kills_per_round'] - df['team_avg_kills']
        
        return df
    
    def prepare_training_data(self, historical_kill_lines: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical kill line data
        
        Args:
            historical_kill_lines: List of dictionaries containing:
                - player_name: str
                - team: str
                - opponent_team: str
                - kill_line: float (the predicted kill line)
                - actual_kills: int (actual kills achieved)
                - map: str (optional)
                - tournament: str (optional)
                - date: str (optional)
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Load player stats
        players_df = self.load_all_players()
        players_df = self.create_features(players_df)
        
        # Create training dataset
        training_data = []
        labels = []
        
        for record in historical_kill_lines:
            player_stats = players_df[
                (players_df['name'] == record['player_name']) & 
                (players_df['team'] == record['team'])
            ]
            
            if not player_stats.empty:
                player_row = player_stats.iloc[0]
                
                # Create feature vector
                features = {
                    'rating': player_row['rating'],
                    'average_combat_score': player_row['average_combat_score'],
                    'kill_deaths': player_row['kill_deaths'],
                    'kills_per_round': player_row['kills_per_round'],
                    'assists_per_round': player_row['assists_per_round'],
                    'first_kills_per_round': player_row['first_kills_per_round'],
                    'first_deaths_per_round': player_row['first_deaths_per_round'],
                    'headshot_percentage': player_row['headshot_percentage'],
                    'clutch_success_percentage': player_row['clutch_success_percentage'],
                    'total_impact': player_row['total_impact'],
                    'survival_rate': player_row['survival_rate'],
                    'efficiency': player_row['efficiency'],
                    'team_avg_rating': player_row['team_avg_rating'],
                    'team_avg_kills': player_row['team_avg_kills'],
                    'relative_rating': player_row['relative_rating'],
                    'relative_kills': player_row['relative_kills'],
                    'kill_line': record['kill_line']
                }
                
                training_data.append(features)
                
                # Create label: 0=under, 1=over, 2=unsure (if confidence is low)
                actual_kills = record['actual_kills']
                kill_line = record['kill_line']
                
                if actual_kills > kill_line:
                    labels.append(1)  # Over
                elif actual_kills < kill_line:
                    labels.append(0)  # Under
                else:
                    labels.append(2)  # Push/Unsure
        
        return pd.DataFrame(training_data), pd.Series(labels) 