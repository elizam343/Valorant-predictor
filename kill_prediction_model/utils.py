import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .models import PredictionResult, PredictionType

class ModelEvaluator:
    """Utilities for evaluating model performance"""
    
    @staticmethod
    def calculate_roi(predictions: List[PredictionResult], 
                     actual_results: List[Dict], 
                     bet_amount: float = 100) -> Dict:
        """
        Calculate Return on Investment for predictions
        
        Args:
            predictions: List of prediction results
            actual_results: List of actual outcomes with 'player_name', 'actual_kills', 'kill_line'
            bet_amount: Amount bet on each prediction
            
        Returns:
            Dictionary with ROI metrics
        """
        total_bets = 0
        total_winnings = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Create lookup for actual results
        actual_lookup = {result['player_name']: result for result in actual_results}
        
        for pred in predictions:
            if pred.player_name not in actual_lookup:
                continue
                
            actual = actual_lookup[pred.player_name]
            actual_kills = actual['actual_kills']
            kill_line = actual['kill_line']
            
            total_predictions += 1
            
            # Determine if prediction was correct
            if pred.prediction == PredictionType.OVER and actual_kills > kill_line:
                correct_predictions += 1
                total_winnings += bet_amount * 1.9  # Typical over/under odds
            elif pred.prediction == PredictionType.UNDER and actual_kills < kill_line:
                correct_predictions += 1
                total_winnings += bet_amount * 1.9
            elif pred.prediction == PredictionType.UNSURE:
                continue  # No bet placed
            else:
                total_winnings += 0  # Lost bet
            
            total_bets += bet_amount
        
        if total_bets == 0:
            return {
                'roi': 0,
                'accuracy': 0,
                'total_bets': 0,
                'total_winnings': 0,
                'profit': 0
            }
        
        roi = (total_winnings - total_bets) / total_bets
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'roi': roi,
            'accuracy': accuracy,
            'total_bets': total_bets,
            'total_winnings': total_winnings,
            'profit': total_winnings - total_bets,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    @staticmethod
    def plot_prediction_distribution(predictions: List[PredictionResult], 
                                   save_path: str = None):
        """Plot distribution of predictions and confidence levels"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prediction types
        pred_types = [p.prediction.name for p in predictions]
        pred_counts = pd.Series(pred_types).value_counts()
        axes[0, 0].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Prediction Distribution')
        
        # Confidence distribution
        confidences = [p.confidence for p in predictions]
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Confidence Distribution')
        
        # Kill lines vs confidence
        kill_lines = [p.kill_line for p in predictions]
        axes[1, 0].scatter(kill_lines, confidences, alpha=0.6)
        axes[1, 0].set_xlabel('Kill Line')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_title('Kill Line vs Confidence')
        
        # Over/Under probabilities
        over_probs = [p.over_probability for p in predictions]
        under_probs = [p.under_probability for p in predictions]
        axes[1, 1].scatter(over_probs, under_probs, alpha=0.6)
        axes[1, 1].set_xlabel('Over Probability')
        axes[1, 1].set_ylabel('Under Probability')
        axes[1, 1].set_title('Over vs Under Probabilities')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

class DataAnalyzer:
    """Utilities for analyzing player and team data"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def get_top_players(self, stat: str = 'kills_per_round', 
                       top_n: int = 10) -> pd.DataFrame:
        """Get top players by a specific statistic"""
        df = self.data_loader.load_all_players()
        df = self.data_loader.create_features(df)
        
        if stat not in df.columns:
            raise ValueError(f"Statistic '{stat}' not found in data")
        
        return df.nlargest(top_n, stat)[['name', 'team', stat]]
    
    def get_team_comparison(self, team1: str, team2: str) -> Dict:
        """Compare two teams across key metrics"""
        df = self.data_loader.load_all_players()
        df = self.data_loader.create_features(df)
        
        team1_data = df[df['team'] == team1]
        team2_data = df[df['team'] == team2]
        
        if team1_data.empty or team2_data.empty:
            raise ValueError(f"One or both teams not found: {team1}, {team2}")
        
        comparison = {}
        metrics = ['rating', 'kills_per_round', 'average_combat_score', 
                  'headshot_percentage', 'clutch_success_percentage']
        
        for metric in metrics:
            if metric in df.columns:
                team1_avg = team1_data[metric].mean()
                team2_avg = team2_data[metric].mean()
                comparison[metric] = {
                    team1: team1_avg,
                    team2: team2_avg,
                    'difference': team1_avg - team2_avg
                }
        
        return comparison
    
    def plot_player_performance(self, player_name: str, team: str = None,
                              save_path: str = None):
        """Create a radar chart of player performance"""
        player_stats = self.data_loader.get_player_stats(player_name, team)
        
        if not player_stats:
            raise ValueError(f"Player {player_name} not found")
        
        # Get all players for percentile calculation
        all_players = self.data_loader.load_all_players()
        all_players = self.data_loader.create_features(all_players)
        
        # Calculate percentiles for key stats
        stats = ['rating', 'kills_per_round', 'average_combat_score', 
                'headshot_percentage', 'clutch_success_percentage']
        percentiles = []
        
        for stat in stats:
            if stat in all_players.columns:
                player_value = getattr(player_stats, stat)
                percentile = (all_players[stat] <= player_value).mean() * 100
                percentiles.append(percentile)
            else:
                percentiles.append(0)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
        percentiles += percentiles[:1]  # Close the plot
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, percentiles, 'o-', linewidth=2, label=player_name)
        ax.fill(angles, percentiles, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(stats)
        ax.set_ylim(0, 100)
        ax.set_title(f'{player_name} Performance Percentiles')
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Player performance plot saved to {save_path}")
        
        plt.show()

class HistoricalDataManager:
    """Manage historical kill line data for training"""
    
    def __init__(self, data_file: str = "historical_data.json"):
        self.data_file = data_file
        self.historical_data = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """Load historical data from file"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_data(self):
        """Save historical data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2)
    
    def add_match_result(self, match_data: Dict):
        """Add a new match result to historical data"""
        required_fields = ['player_name', 'team', 'opponent_team', 
                          'kill_line', 'actual_kills']
        
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Missing required field: {field}")
        
        match_data['date_added'] = datetime.now().isoformat()
        self.historical_data.append(match_data)
        self.save_data()
    
    def get_recent_matches(self, days: int = 30) -> List[Dict]:
        """Get matches from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_matches = []
        
        for match in self.historical_data:
            if 'date' in match:
                try:
                    match_date = datetime.fromisoformat(match['date'])
                    if match_date >= cutoff_date:
                        recent_matches.append(match)
                except ValueError:
                    continue
        
        return recent_matches
    
    def get_player_history(self, player_name: str, team: str = None) -> List[Dict]:
        """Get historical data for a specific player"""
        player_matches = []
        
        for match in self.historical_data:
            if match['player_name'] == player_name:
                if team is None or match['team'] == team:
                    player_matches.append(match)
        
        return sorted(player_matches, key=lambda x: x.get('date', ''), reverse=True)
    
    def calculate_player_stats(self, player_name: str, team: str = None) -> Dict:
        """Calculate historical performance stats for a player"""
        matches = self.get_player_history(player_name, team)
        
        if not matches:
            return {"error": f"No historical data found for {player_name}"}
        
        total_matches = len(matches)
        over_hits = sum(1 for m in matches if m['actual_kills'] > m['kill_line'])
        under_hits = sum(1 for m in matches if m['actual_kills'] < m['kill_line'])
        pushes = sum(1 for m in matches if m['actual_kills'] == m['kill_line'])
        
        avg_kills = np.mean([m['actual_kills'] for m in matches])
        avg_kill_line = np.mean([m['kill_line'] for m in matches])
        
        return {
            'player_name': player_name,
            'team': team,
            'total_matches': total_matches,
            'over_hits': over_hits,
            'under_hits': under_hits,
            'pushes': pushes,
            'over_rate': over_hits / total_matches,
            'under_rate': under_hits / total_matches,
            'push_rate': pushes / total_matches,
            'avg_actual_kills': avg_kills,
            'avg_kill_line': avg_kill_line,
            'avg_difference': avg_kills - avg_kill_line
        }

def create_sample_historical_dataset() -> List[Dict]:
    """Create a comprehensive sample dataset for testing"""
    return [
        {
            'player_name': 'TenZ',
            'team': 'Sentinels',
            'opponent_team': 'Cloud9',
            'kill_line': 18.5,
            'actual_kills': 22,
            'map': 'Ascent',
            'tournament': 'VCT Champions',
            'date': '2024-01-15'
        },
        {
            'player_name': 'TenZ',
            'team': 'Sentinels',
            'opponent_team': 'Team Liquid',
            'kill_line': 16.5,
            'actual_kills': 14,
            'map': 'Haven',
            'tournament': 'VCT Champions',
            'date': '2024-01-16'
        },
        {
            'player_name': 'ShahZaM',
            'team': 'Sentinels',
            'opponent_team': 'Cloud9',
            'kill_line': 15.5,
            'actual_kills': 18,
            'map': 'Ascent',
            'tournament': 'VCT Champions',
            'date': '2024-01-15'
        },
        {
            'player_name': 'ShahZaM',
            'team': 'Sentinels',
            'opponent_team': 'Team Liquid',
            'kill_line': 14.5,
            'actual_kills': 12,
            'map': 'Haven',
            'tournament': 'VCT Champions',
            'date': '2024-01-16'
        }
    ]

def export_sample_data():
    """Export sample historical data to JSON file"""
    sample_data = create_sample_historical_dataset()
    
    with open('sample_historical_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample historical data exported to sample_historical_data.json")
    return sample_data 