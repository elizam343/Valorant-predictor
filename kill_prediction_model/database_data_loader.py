"""
Database-based data loader for ML model training
Uses SQLite database instead of individual JSON files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseDataLoader:
    def __init__(self, db_path: str = "../Scraper/valorant_matches.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def load_player_match_data(self, min_matches: int = 5, days_back: int = 365) -> pd.DataFrame:
        """Load player match data from database"""
        logger.info("Loading player match data from database...")
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT 
            p.name as player_name,
            t.name as team_name,
            pms.team_id as team_id,
            m.match_date,
            m.series_type,
            tour.name as tournament_name,
            mp.map_name,
            pms.kills,
            pms.deaths,
            pms.assists,
            pms.acs,
            pms.adr,
            pms.fk,
            pms.hs_percentage,
            pms.kdr,
            m.match_id
        FROM player_match_stats pms
        JOIN players p ON pms.player_id = p.id
        JOIN teams t ON pms.team_id = t.id
        JOIN matches m ON pms.match_id = m.id
        JOIN maps mp ON pms.map_id = mp.id
        JOIN tournaments tour ON m.tournament_id = tour.id
        WHERE m.match_date >= ?
        ORDER BY m.match_date DESC
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        
        logger.info(f"Loaded {len(df)} player-match records")
        
        # Filter players with minimum matches
        player_match_counts = df['player_name'].value_counts()
        valid_players = player_match_counts[player_match_counts >= min_matches].index
        df = df[df['player_name'].isin(valid_players)]
        
        logger.info(f"Filtered to {len(df)} records from {len(valid_players)} players")
        
        return df
    
    def calculate_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced features for each player-match (enhanced and scalable)"""
        logger.info("Calculating player features (enhanced, scalable)...")

        # Convert date to datetime
        df['match_date'] = pd.to_datetime(df['match_date'])
        # Sort by player and date
        df = df.sort_values(['player_name', 'match_date'])

        # --- Add opponent_team column ---
        # Assumes df has columns: match_id, team_id, and team_name
        # For each match, map team_id to team_name, then assign opponent_team as the other team in the match
        if 'opponent_team' not in df.columns:
            # Get mapping from team_id to team_name
            team_id_to_name = df[['team_id', 'team_name']].drop_duplicates().set_index('team_id')['team_name'].to_dict()
            # For each match, get the two team_ids
            match_team_ids = df.groupby('match_id')['team_id'].unique().to_dict()
            # For each row, assign opponent_team as the other team in the match
            def get_opponent(row):
                teams = match_team_ids.get(row['match_id'], [])
                if len(teams) == 2:
                    opp_id = teams[0] if teams[1] == row['team_id'] else teams[1]
                    return team_id_to_name.get(opp_id, 'Unknown')
                return 'Unknown'
            df['opponent_team'] = df.apply(get_opponent, axis=1)

        # Rolling averages for each player (last 10 matches)
        df['avg_kills_10'] = (
            df.groupby('player_name')['kills']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['avg_deaths_10'] = (
            df.groupby('player_name')['deaths']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['avg_assists_10'] = (
            df.groupby('player_name')['assists']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['avg_acs_10'] = (
            df.groupby('player_name')['acs']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['avg_adr_10'] = (
            df.groupby('player_name')['adr']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['avg_kdr_10'] = (
            df.groupby('player_name')['kdr']
              .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        )

        # Recent form (last 5 matches)
        df['recent_kills_5'] = (
            df.groupby('player_name')['kills']
              .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['recent_acs_5'] = (
            df.groupby('player_name')['acs']
              .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df['recent_kdr_5'] = (
            df.groupby('player_name')['kdr']
              .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )

        # Map-specific player stats (average kills/acs/kdr on each map)
        map_player_stats = df.groupby(['player_name', 'map_name']).agg({
            'kills': 'mean',
            'acs': 'mean',
            'kdr': 'mean'
        }).reset_index()
        map_player_stats.columns = ['player_name', 'map_name', 'map_avg_kills', 'map_avg_acs', 'map_avg_kdr']
        df = df.merge(map_player_stats, on=['player_name', 'map_name'], how='left')

        # Opponent-specific stats (player's average kills/acs/kdr vs. current opponent)
        opp_stats = df.groupby(['player_name', 'opponent_team']).agg({
            'kills': 'mean',
            'acs': 'mean',
            'kdr': 'mean'
        }).reset_index()
        opp_stats.columns = ['player_name', 'opponent_team', 'opp_avg_kills', 'opp_avg_acs', 'opp_avg_kdr']
        df = df.merge(opp_stats, on=['player_name', 'opponent_team'], how='left')

        # Team strength features
        team_stats = df.groupby('team_name').agg({
            'kills': 'mean',
            'acs': 'mean',
            'kdr': 'mean'
        }).reset_index()
        team_stats.columns = ['team_name', 'team_avg_kills', 'team_avg_acs', 'team_avg_kdr']
        df = df.merge(team_stats, on='team_name', how='left')

        # Tournament importance (ordinal feature)
        tournament_importance = {
            'VCT Champions': 5,
            'VCT Masters': 4,
            'VCT International': 4,
            'Regional': 3,
            'Qualifier': 2,
            'Showmatch': 1
        }
        df['tournament_importance'] = df['tournament_name'].map(
            lambda x: next((v for k, v in tournament_importance.items() if k.lower() in x.lower()), 2)
        )

        # Series type importance
        series_importance = {'bo1': 1, 'bo3': 2, 'bo5': 3}
        df['series_importance'] = df['series_type'].map(series_importance)

        # Days since last match
        df['days_since_last_match'] = df.groupby('player_name')['match_date'].diff().dt.days

        # Fill NaN values
        df = df.fillna(0)

        logger.info(f"Calculated features for {len(df)} records (enhanced, scalable)")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        # Select features for training
        feature_columns = [
            'avg_kills_10', 'avg_deaths_10', 'avg_assists_10', 'avg_acs_10', 'avg_adr_10', 'avg_kdr_10',
            'recent_kills_5', 'recent_acs_5', 'recent_kdr_5',
            'team_avg_kills', 'team_avg_acs', 'team_avg_kdr',
            'map_avg_kills', 'map_avg_acs',
            'tournament_importance', 'series_importance', 'days_since_last_match'
        ]
        
        # Target variable: kills per round (simplified as kills for now)
        target_column = 'kills'
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        logger.info(f"Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y, feature_columns
    
    def get_player_context(self, player_name: str, team_name: str, opponent_team: str, 
                          tournament: str, series_type: str, maps: List[str]) -> np.ndarray:
        """Get context features for a specific matchup"""
        logger.info(f"Getting context for {player_name} vs {opponent_team}")
        
        with self.get_connection() as conn:
            # Get player's recent stats
            player_query = """
            SELECT 
                pms.kills, pms.deaths, pms.assists, pms.acs, pms.adr, pms.kdr
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.id
            JOIN matches m ON pms.match_id = m.id
            WHERE p.name = ?
            ORDER BY m.match_date DESC
            LIMIT 10
            """
            
            player_stats = pd.read_sql_query(player_query, conn, params=(player_name,))
            
            if len(player_stats) == 0:
                logger.warning(f"No data found for player {player_name}")
                return None
            
            # Calculate averages
            avg_kills = player_stats['kills'].mean()
            avg_deaths = player_stats['deaths'].mean()
            avg_assists = player_stats['assists'].mean()
            avg_acs = player_stats['acs'].mean()
            avg_adr = player_stats['adr'].mean()
            avg_kdr = player_stats['kdr'].mean()
            
            # Recent form (last 5 matches)
            recent_kills = player_stats['kills'].head(5).mean()
            recent_acs = player_stats['acs'].head(5).mean()
            recent_kdr = player_stats['kdr'].head(5).mean()
            
            # Team strength
            team_query = """
            SELECT AVG(pms.kills) as team_avg_kills, AVG(pms.acs) as team_avg_acs, AVG(pms.kdr) as team_avg_kdr
            FROM player_match_stats pms
            JOIN teams t ON pms.team_id = t.id
            JOIN matches m ON pms.match_id = m.id
            WHERE t.name = ?
            AND m.match_date >= date('now', '-30 days')
            """
            
            team_stats = pd.read_sql_query(team_query, conn, params=(team_name,))
            team_avg_kills = team_stats['team_avg_kills'].iloc[0] if len(team_stats) > 0 else 0
            team_avg_acs = team_stats['team_avg_acs'].iloc[0] if len(team_stats) > 0 else 0
            team_avg_kdr = team_stats['team_avg_kdr'].iloc[0] if len(team_stats) > 0 else 0
            
            # Map familiarity
            map_familiarity = []
            for map_name in maps:
                map_query = """
                SELECT AVG(pms.kills) as map_avg_kills, AVG(pms.acs) as map_avg_acs
                FROM player_match_stats pms
                JOIN players p ON pms.player_id = p.id
                JOIN maps mp ON pms.map_id = mp.id
                WHERE p.name = ? AND mp.map_name = ?
                """
                
                map_stats = pd.read_sql_query(map_query, conn, params=(player_name, map_name))
                map_avg_kills = map_stats['map_avg_kills'].iloc[0] if len(map_stats) > 0 else 0
                map_avg_acs = map_stats['map_avg_acs'].iloc[0] if len(map_stats) > 0 else 0
                map_familiarity.extend([map_avg_kills, map_avg_acs])
            
            # Use average of map familiarity if multiple maps
            if len(maps) > 1:
                map_avg_kills = np.mean([map_familiarity[i] for i in range(0, len(map_familiarity), 2)])
                map_avg_acs = np.mean([map_familiarity[i] for i in range(1, len(map_familiarity), 2)])
                map_familiarity = [map_avg_kills, map_avg_acs]
            
            # Tournament importance
            tournament_importance = {
                'VCT Champions': 5, 'VCT Masters': 4, 'VCT International': 4,
                'Regional': 3, 'Qualifier': 2, 'Showmatch': 1
            }
            tourn_importance = next((v for k, v in tournament_importance.items() 
                                   if k.lower() in tournament.lower()), 2)
            
            # Series type importance
            series_importance = {'bo1': 1, 'bo3': 2, 'bo5': 3}
            series_imp = series_importance.get(series_type.lower(), 1)
            
            # Days since last match
            last_match_query = """
            SELECT MAX(m.match_date) as last_match
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.id
            JOIN matches m ON pms.match_id = m.id
            WHERE p.name = ?
            """
            
            last_match = pd.read_sql_query(last_match_query, conn, params=(player_name,))
            days_since_last = 0
            if len(last_match) > 0 and last_match['last_match'].iloc[0]:
                last_date = pd.to_datetime(last_match['last_match'].iloc[0])
                days_since_last = (datetime.now() - last_date).days
        
        # Create feature vector
        features = [
            avg_kills, avg_deaths, avg_assists, avg_acs, avg_adr, avg_kdr,
            recent_kills, recent_acs, recent_kdr,
            team_avg_kills, team_avg_acs, team_avg_kdr,
            map_familiarity[0], map_familiarity[1],  # map_avg_kills, map_avg_acs
            tourn_importance, series_imp, days_since_last
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        return features_scaled[0]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            
            # Count records
            for table in ['matches', 'players', 'teams', 'player_match_stats']:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Date range
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(match_date), MAX(match_date) FROM matches")
            result = cursor.fetchone()
            stats['date_range'] = {'earliest': result[0], 'latest': result[1]}
            
            return stats

def main():
    """Test the database data loader"""
    loader = DatabaseDataLoader()
    
    # Load data
    df = loader.load_player_match_data(min_matches=5, days_back=365)
    
    # Calculate features
    feature_df = loader.calculate_player_features(df)
    
    # Prepare training data
    X, y, feature_columns = loader.prepare_training_data(feature_df)
    
    print(f"Training data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")
    
    # Test player context
    context = loader.get_player_context(
        player_name="aspas",
        team_name="MIBR",
        opponent_team="FUR",
        tournament="VCT Champions",
        series_type="bo3",
        maps=["Ascent", "Haven"]
    )
    
    if context is not None:
        print(f"Context features shape: {context.shape}")
    
    # Database stats
    stats = loader.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 