import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from .data_loader import DataLoader, PlayerStats
from .models import KillLinePredictor, EnsemblePredictor, PredictionResult, PredictionType

@dataclass
class KillLineBet:
    """Represents a kill line betting opportunity"""
    player_name: str
    team: str
    opponent_team: str
    kill_line: float
    map: str = "Unknown"
    tournament: str = "Unknown"
    match_date: str = "Unknown"

class ValorantKillPredictor:
    """Main interface for Valorant kill line predictions"""
    
    def __init__(self, model_path: str = None, use_ensemble: bool = True):
        self.data_loader = DataLoader()
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            self.predictor = EnsemblePredictor()
        else:
            self.predictor = KillLinePredictor()
        
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, historical_data: List[Dict], save_path: str = None):
        """
        Train the prediction model using historical kill line data
        
        Args:
            historical_data: List of dictionaries with historical kill line data
            save_path: Optional path to save the trained model
        """
        print("Preparing training data...")
        X, y = self.data_loader.prepare_training_data(historical_data)
        
        print(f"Training data shape: {X.shape}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Train the model
        if self.use_ensemble:
            results = self.predictor.train(X, y)
        else:
            results = self.predictor.train(X, y)
        
        # Save model if path provided
        if save_path:
            if self.use_ensemble:
                self.predictor.save_ensemble(save_path)
            else:
                self.predictor.save_model(save_path)
        
        return results
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        if self.use_ensemble:
            self.predictor.load_ensemble(model_path)
        else:
            self.predictor.load_model(model_path)
    
    def predict_kill_line(self, bet: KillLineBet) -> PredictionResult:
        """
        Predict whether a player will go over/under their kill line
        
        Args:
            bet: KillLineBet object containing player and match information
            
        Returns:
            PredictionResult with prediction and confidence
        """
        # Get player stats
        player_stats = self.data_loader.get_player_stats(bet.player_name, bet.team)
        
        if not player_stats:
            raise ValueError(f"Player {bet.player_name} from team {bet.team} not found in database")
        
        # Create feature vector
        players_df = self.data_loader.load_all_players()
        players_df = self.data_loader.create_features(players_df)
        
        player_row = players_df[
            (players_df['name'] == bet.player_name) & 
            (players_df['team'] == bet.team)
        ].iloc[0]
        
        features = {
            'player_name': bet.player_name,
            'team': bet.team,
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
            'kill_line': bet.kill_line
        }
        
        # Make prediction
        return self.predictor.predict(features)
    
    def predict_multiple_bets(self, bets: List[KillLineBet]) -> List[PredictionResult]:
        """Predict multiple kill line bets at once"""
        return [self.predict_kill_line(bet) for bet in bets]
    
    def get_high_confidence_picks(self, bets: List[KillLineBet], 
                                 min_confidence: float = 0.7) -> List[PredictionResult]:
        """Get only high-confidence predictions"""
        all_predictions = self.predict_multiple_bets(bets)
        return [pred for pred in all_predictions if pred.confidence >= min_confidence]
    
    def analyze_player_history(self, player_name: str, team: str = None) -> Dict:
        """Analyze a player's historical performance"""
        player_stats = self.data_loader.get_player_stats(player_name, team)
        
        if not player_stats:
            return {"error": f"Player {player_name} not found"}
        
        # Get all players for comparison
        all_players = self.data_loader.load_all_players()
        all_players = self.data_loader.create_features(all_players)
        
        # Calculate percentiles
        percentiles = {}
        for stat in ['rating', 'kills_per_round', 'average_combat_score', 'headshot_percentage']:
            if stat in all_players.columns:
                player_value = getattr(player_stats, stat)
                percentile = (all_players[stat] <= player_value).mean() * 100
                percentiles[stat] = percentile
        
        return {
            "player_name": player_name,
            "team": player_stats.team,
            "stats": {
                "rating": player_stats.rating,
                "kills_per_round": player_stats.kills_per_round,
                "average_combat_score": player_stats.average_combat_score,
                "headshot_percentage": player_stats.headshot_percentage,
                "clutch_success_percentage": player_stats.clutch_success_percentage
            },
            "percentiles": percentiles,
            "strengths": [stat for stat, pct in percentiles.items() if pct >= 75],
            "weaknesses": [stat for stat, pct in percentiles.items() if pct <= 25]
        }
    
    def generate_betting_report(self, bets: List[KillLineBet], 
                              output_file: str = None) -> str:
        """Generate a comprehensive betting report"""
        predictions = self.predict_multiple_bets(bets)
        
        # Categorize predictions
        over_picks = [p for p in predictions if p.prediction == PredictionType.OVER]
        under_picks = [p for p in predictions if p.prediction == PredictionType.UNDER]
        unsure_picks = [p for p in predictions if p.prediction == PredictionType.UNSURE]
        
        # High confidence picks
        high_confidence = [p for p in predictions if p.confidence >= 0.7]
        
        # Generate report
        report = f"""
VALORANT KILL LINE PREDICTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Bets Analyzed: {len(bets)}

PREDICTION SUMMARY:
- Over Picks: {len(over_picks)}
- Under Picks: {len(under_picks)}
- Unsure/Avoid: {len(unsure_picks)}
- High Confidence Picks: {len(high_confidence)}

RECOMMENDED BETS (High Confidence):
"""
        
        for pred in sorted(high_confidence, key=lambda x: x.confidence, reverse=True):
            report += f"""
{pred.player_name} ({pred.team})
Kill Line: {pred.kill_line}
Prediction: {pred.recommended_action}
Confidence: {pred.confidence:.1%}
Over Probability: {pred.over_probability:.1%}
Under Probability: {pred.under_probability:.1%}
"""
        
        if over_picks:
            report += "\nOVER PICKS:\n"
            for pred in sorted(over_picks, key=lambda x: x.confidence, reverse=True):
                report += f"- {pred.player_name}: {pred.confidence:.1%} confidence\n"
        
        if under_picks:
            report += "\nUNDER PICKS:\n"
            for pred in sorted(under_picks, key=lambda x: x.confidence, reverse=True):
                report += f"- {pred.player_name}: {pred.confidence:.1%} confidence\n"
        
        # Save report if file path provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report
    
    def export_predictions_csv(self, bets: List[KillLineBet], 
                             output_file: str = "predictions.csv"):
        """Export predictions to CSV format"""
        predictions = self.predict_multiple_bets(bets)
        
        data = []
        for pred in predictions:
            data.append({
                'player_name': pred.player_name,
                'team': pred.team,
                'kill_line': pred.kill_line,
                'prediction': pred.prediction.name,
                'confidence': pred.confidence,
                'over_probability': pred.over_probability,
                'under_probability': pred.under_probability,
                'unsure_probability': pred.unsure_probability,
                'recommended_action': pred.recommended_action
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Predictions exported to {output_file}")
        return df

# Example usage functions
def create_sample_historical_data() -> List[Dict]:
    """Create sample historical data for training"""
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
        }
        # Add more historical data here
    ]

def example_usage():
    """Example of how to use the prediction system"""
    # Initialize predictor
    predictor = ValorantKillPredictor(use_ensemble=True)
    
    # Train model (if you have historical data)
    # historical_data = create_sample_historical_data()
    # predictor.train_model(historical_data, save_path="models/kill_predictor.pkl")
    
    # Load pre-trained model
    # predictor.load_model("models/kill_predictor.pkl")
    
    # Create betting opportunities
    bets = [
        KillLineBet("TenZ", "Sentinels", "Cloud9", 18.5, "Ascent", "VCT Champions"),
        KillLineBet("ShahZaM", "Sentinels", "Cloud9", 15.5, "Ascent", "VCT Champions"),
    ]
    
    # Make predictions
    predictions = predictor.predict_multiple_bets(bets)
    
    # Generate report
    report = predictor.generate_betting_report(bets, "betting_report.txt")
    print(report)
    
    return predictions 