#!/usr/bin/env python3
"""
Kill Prediction Script
Predicts kills per round for a player and compares to kill lines
"""

import joblib
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
from gpu_trainer import KillPredictionNN

class KillPredictor:
    def __init__(self, model_path: str = "models/neural_network_gpu_model.pkl"):
        """Initialize the kill predictor with a trained model"""
        try:
            self.model_data = joblib.load(model_path)
            
            # Check if it's a GPU model
            if 'model_state_dict' in self.model_data:
                # GPU model
                # Use the saved model's architecture
                input_size = self.model_data['input_size']
                hidden_sizes = self.model_data['hidden_sizes']
                self.model = KillPredictionNN(input_size, hidden_sizes=hidden_sizes).to('cpu')
                self.model.load_state_dict(self.model_data['model_state_dict'])
                self.model.eval()
                self.scaler = self.model_data['scaler']
                self.feature_columns = self.model_data['feature_columns']
                self.model_type = 'gpu'
            else:
                # CPU model
                self.model = self.model_data['model']
                self.scaler = self.model_data['scaler']
                self.feature_columns = self.model_data['feature_columns']
                self.model_type = 'cpu'
            
            print(f"Loaded {self.model_type} model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.model_type = None
    
    def predict_kills_per_round(self, player_stats: Dict) -> Tuple[float, float]:
        """
        Predict kills per round for a player based on their stats
        
        Args:
            player_stats: Dictionary containing player statistics
            
        Returns:
            Tuple of (predicted_kills_per_round, confidence_score)
        """
        if self.model is None:
            return None, None
        
        # Create feature vector
        features = []
        for col in self.feature_columns:
            if col in player_stats:
                features.append(player_stats[col])
            else:
                # Use default values for missing features
                features.append(self._get_default_value(col))
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        if self.model_type == 'gpu':
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                predicted_kills = self.model(features_tensor).item()
        else:
            predicted_kills = self.model.predict(features_scaled)[0]
        
        # Calculate confidence (simplified - could be improved)
        confidence = 0.7  # Placeholder confidence
        
        return predicted_kills, confidence
    
    def compare_to_kill_line(self, predicted_kills_per_round: float, kill_line: float, confidence: float = 0.7) -> Dict:
        """
        Compare predicted kills per round to a kill line and provide recommendation
        
        Args:
            predicted_kills_per_round: Model's prediction of kills per round
            kill_line: Bookmaker's kill line (kills per round)
            confidence: Model confidence (0-1)
            
        Returns:
            Dictionary with recommendation and analysis
        """
        difference = predicted_kills_per_round - kill_line
        percentage_diff = (difference / kill_line) * 100 if kill_line > 0 else 0
        
        # Determine recommendation based on confidence threshold
        confidence_threshold = 0.90  # 90% threshold as specified
        
        if confidence >= confidence_threshold:
            if difference > 0:
                recommendation = "OVER"
                reasoning = f"Model predicts {predicted_kills_per_round:.3f} kills/round vs {kill_line:.3f} line (+{difference:.3f})"
            else:
                recommendation = "UNDER"
                reasoning = f"Model predicts {predicted_kills_per_round:.3f} kills/round vs {kill_line:.3f} line ({difference:.3f})"
        else:
            recommendation = "UNSURE"
            reasoning = f"Low confidence ({confidence:.1%}) - model predicts {predicted_kills_per_round:.3f} kills/round vs {kill_line:.3f} line"
        
        return {
            'predicted_kills_per_round': predicted_kills_per_round,
            'kill_line': kill_line,
            'difference': difference,
            'percentage_diff': percentage_diff,
            'confidence': confidence,
            'recommendation': recommendation,
            'reasoning': reasoning
        }
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default values for missing features"""
        defaults = {
            'db_rating': 1.0,
            'db_average_combat_score': 200.0,
            'db_kill_deaths': 1.0,
            'db_kills_per_round': 0.8,
            'db_assists_per_round': 0.3,
            'db_first_kills_per_round': 0.1,
            'db_first_deaths_per_round': 0.1,
            'db_headshot_percentage': 0.25,
            'db_clutch_success_percentage': 0.5,
            'opponent_team_strength': 1.0,
            'team_strength': 1.0,
            'map_familiarity': 0.5,
            'recent_form': 1.0,
            'tournament_importance': 0.5
        }
        return defaults.get(feature_name, 0.0)

def main():
    """Example usage of the kill predictor"""
    predictor = KillPredictor()
    
    if predictor.model is None:
        print("No model available. Please train a model first.")
        return
    
    # Example player stats (these would come from your database)
    example_player = {
        'db_rating': 1.15,
        'db_average_combat_score': 245.0,
        'db_kill_deaths': 1.2,
        'db_kills_per_round': 0.85,
        'db_assists_per_round': 0.35,
        'db_first_kills_per_round': 0.12,
        'db_first_deaths_per_round': 0.08,
        'db_headshot_percentage': 0.28,
        'db_clutch_success_percentage': 0.55,
        'opponent_team_strength': 1.1,
        'team_strength': 1.05,
        'map_familiarity': 0.7,
        'recent_form': 1.1,
        'tournament_importance': 0.8
    }
    
    # Example kill lines to test (kills per round)
    kill_lines = [0.75, 0.85, 0.95, 1.05]
    
    print("=== Kill Line Prediction Example (Kills per Round) ===\n")
    
    # Predict kills per round
    predicted_kills_per_round, confidence = predictor.predict_kills_per_round(example_player)
    
    if predicted_kills_per_round is not None:
        print(f"Player Stats Summary:")
        print(f"- Rating: {example_player['db_rating']:.2f}")
        print(f"- ACS: {example_player['db_average_combat_score']:.0f}")
        print(f"- K/D: {example_player['db_kill_deaths']:.2f}")
        print(f"- Kills per round: {example_player['db_kills_per_round']:.2f}")
        print(f"- Recent form: {example_player['recent_form']:.2f}")
        print(f"\nModel Prediction: {predicted_kills_per_round:.3f} kills/round (confidence: {confidence:.1%})")
        
        print(f"\n=== Kill Line Analysis ===")
        for kill_line in kill_lines:
            result = predictor.compare_to_kill_line(predicted_kills_per_round, kill_line, confidence)
            print(f"\nKill Line: {kill_line:.3f} kills/round")
            print(f"Recommendation: {result['recommendation']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Difference: {result['difference']:+.3f} ({result['percentage_diff']:+.1f}%)")
    else:
        print("Failed to make prediction")

if __name__ == "__main__":
    main() 