#!/usr/bin/env python3
"""
Advanced Matchup Predictor
Predicts kills per round for specific matchups with confidence intervals
"""

import joblib
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from scipy import stats
from dataclasses import dataclass
import json
from gpu_trainer import KillPredictionNN
from enhanced_data_loader import EnhancedDataLoader

@dataclass
class MatchupContext:
    """Context for a specific matchup prediction"""
    player_name: str
    player_team: str
    opponent_team: str
    tournament: str
    series_type: str  # e.g., "bo3", "bo5", "group_stage", "playoffs"
    maps: List[str]
    kill_line: float  # bookmaker's kill line (kills per round)
    
    def to_dict(self) -> Dict:
        return {
            'player_name': self.player_name,
            'player_team': self.player_team,
            'opponent_team': self.opponent_team,
            'tournament': self.tournament,
            'series_type': self.series_type,
            'maps': self.maps,
            'kill_line': self.kill_line
        }

@dataclass
class PredictionResult:
    """Result of a matchup prediction"""
    predicted_kills_per_round: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_90: Tuple[float, float]
    confidence_interval_80: Tuple[float, float]
    prediction_std: float
    confidence_score: float  # 0-1, how confident we are in this prediction
    recommendation: str  # "OVER", "UNDER", "UNSURE"
    statistical_significance: float  # p-value for difference from kill line
    effect_size: float  # standardized difference from kill line
    reasoning: str
    
    def to_dict(self) -> Dict:
        return {
            'predicted_kills_per_round': self.predicted_kills_per_round,
            'confidence_interval_95': self.confidence_interval_95,
            'confidence_interval_90': self.confidence_interval_90,
            'confidence_interval_80': self.confidence_interval_80,
            'prediction_std': self.prediction_std,
            'confidence_score': self.confidence_score,
            'recommendation': self.recommendation,
            'statistical_significance': self.statistical_significance,
            'effect_size': self.effect_size,
            'reasoning': self.reasoning
        }

class AdvancedMatchupPredictor:
    def __init__(self, model_path: str = "models/neural_network_gpu_model.pkl"):
        """Initialize the advanced predictor with trained model"""
        self.load_model(model_path)
        self.data_loader = EnhancedDataLoader()
        
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            self.model_data = joblib.load(model_path)
            
            if 'model_state_dict' in self.model_data:
                # GPU model
                input_size = self.model_data['input_size']
                hidden_sizes = self.model_data['hidden_sizes']
                self.model = KillPredictionNN(input_size, hidden_sizes=hidden_sizes).to('cpu')
                self.model.load_state_dict(self.model_data['model_state_dict'])
                self.model.eval()
                self.scaler = self.model_data['scaler']
                self.feature_columns = self.model_data['feature_columns']
                self.model_type = 'gpu'
                print(f"Loaded GPU model with {len(self.feature_columns)} features")
            else:
                raise ValueError("Model format not supported")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.feature_columns = None
    
    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Get player's database statistics"""
        try:
            # Load player database
            player_db = self.data_loader.load_player_database_stats()
            player_data = player_db[player_db['name'] == player_name]
            
            if player_data.empty:
                return None
            
            player_row = player_data.iloc[0]
            return {
                'db_rating': player_row['rating'],
                'db_average_combat_score': player_row['average_combat_score'],
                'db_kill_deaths': player_row['kill_deaths'],
                'db_kills_per_round': player_row['kills_per_round'],
                'db_assists_per_round': player_row['assists_per_round'],
                'db_first_kills_per_round': player_row['first_kills_per_round'],
                'db_first_deaths_per_round': player_row['first_deaths_per_round'],
                'db_headshot_percentage': player_row['headshot_percentage'],
                'db_clutch_success_percentage': player_row['clutch_success_percentage']
            }
        except Exception as e:
            print(f"Error getting player stats: {e}")
            return None
    
    def calculate_team_strength(self, team_name: str) -> float:
        """Calculate team strength based on recent performance"""
        try:
            # Load recent matches to calculate team strength
            matches = self.data_loader.load_scraped_matches(limit=1000)
            
            team_ratings = []
            for match in matches:
                for player in match.players:
                    if player.team == team_name:
                        team_ratings.append(player.rating)
            
            if team_ratings:
                return np.mean(team_ratings)
            else:
                return 1.0  # Default team strength
        except:
            return 1.0
    
    def calculate_map_familiarity(self, player_name: str, maps: List[str]) -> float:
        """Calculate player's familiarity with the maps"""
        try:
            matches = self.data_loader.load_scraped_matches(limit=2000)
            
            player_map_performance = {}
            for match in matches:
                for player in match.players:
                    if player.name == player_name:
                        if player.map_name not in player_map_performance:
                            player_map_performance[player.map_name] = []
                        player_map_performance[player.map_name].append(player.rating)
            
            # Calculate average performance on the specific maps
            map_ratings = []
            for map_name in maps:
                if map_name in player_map_performance:
                    map_ratings.extend(player_map_performance[map_name])
            
            if map_ratings:
                return np.mean(map_ratings) / 1.5  # Normalize to 0-1 range
            else:
                return 0.5  # Default familiarity
        except:
            return 0.5
    
    def calculate_recent_form(self, player_name: str) -> float:
        """Calculate player's recent form based on last 10 matches"""
        try:
            matches = self.data_loader.load_scraped_matches(limit=2000)
            
            player_recent_matches = []
            for match in matches:
                for player in match.players:
                    if player.name == player_name:
                        player_recent_matches.append(player.rating)
                        if len(player_recent_matches) >= 10:
                            break
                if len(player_recent_matches) >= 10:
                    break
            
            if player_recent_matches:
                return np.mean(player_recent_matches) / 1.5  # Normalize
            else:
                return 1.0  # Default form
        except:
            return 1.0
    
    def calculate_tournament_importance(self, tournament: str, series_type: str) -> float:
        """Calculate tournament importance factor"""
        importance_map = {
            'masters': 0.9,
            'champions': 1.0,
            'international': 0.8,
            'regional': 0.6,
            'qualifier': 0.4,
            'showmatch': 0.3
        }
        
        series_importance = {
            'bo5': 1.0,
            'bo3': 0.8,
            'bo1': 0.6,
            'playoffs': 0.9,
            'group_stage': 0.7,
            'finals': 1.0
        }
        
        tournament_importance = importance_map.get(tournament.lower(), 0.5)
        series_importance_val = series_importance.get(series_type.lower(), 0.7)
        
        return (tournament_importance + series_importance_val) / 2
    
    def create_matchup_features(self, matchup: MatchupContext) -> Dict:
        """Create feature vector for the specific matchup"""
        player_stats = self.get_player_stats(matchup.player_name)
        if not player_stats:
            raise ValueError(f"Player {matchup.player_name} not found in database")
        
        # Calculate contextual features
        player_team_strength = self.calculate_team_strength(matchup.player_team)
        opponent_team_strength = self.calculate_team_strength(matchup.opponent_team)
        map_familiarity = self.calculate_map_familiarity(matchup.player_name, matchup.maps)
        recent_form = self.calculate_recent_form(matchup.player_name)
        tournament_importance = self.calculate_tournament_importance(matchup.tournament, matchup.series_type)
        
        features = {
            **player_stats,
            'opponent_team_strength': opponent_team_strength,
            'team_strength': player_team_strength,
            'map_familiarity': map_familiarity,
            'recent_form': recent_form,
            'tournament_importance': tournament_importance
        }
        
        return features
    
    def predict_with_uncertainty(self, features: Dict, n_bootstrap: int = 1000) -> Tuple[float, float, List[float]]:
        """Predict kills per round with uncertainty estimation using bootstrap"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            if col in features:
                feature_vector.append(features[col])
            else:
                feature_vector.append(self._get_default_value(col))
        
        # Scale features
        features_scaled = self.scaler.transform([feature_vector])
        
        # Bootstrap predictions
        predictions = []
        for _ in range(n_bootstrap):
            # Add small noise to features for bootstrap
            noise = np.random.normal(0, 0.01, len(feature_vector))
            noisy_features = np.array(feature_vector) + noise
            noisy_features_scaled = self.scaler.transform([noisy_features])
            
            # Make prediction
            if self.model_type == 'gpu':
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(noisy_features_scaled)
                    pred = self.model(features_tensor).item()
            else:
                pred = self.model.predict(noisy_features_scaled)[0]
            
            predictions.append(pred)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        return mean_prediction, prediction_std, predictions
    
    def calculate_confidence_intervals(self, predictions: List[float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from bootstrap predictions"""
        predictions = np.array(predictions)
        
        intervals = {}
        for confidence in [80, 90, 95]:
            alpha = 1 - confidence / 100
            lower = np.percentile(predictions, alpha/2 * 100)
            upper = np.percentile(predictions, (1 - alpha/2) * 100)
            intervals[f'confidence_interval_{confidence}'] = (float(lower), float(upper))
        
        return intervals
    
    def statistical_comparison(self, predicted_kills: float, kill_line: float, 
                             prediction_std: float, n_samples: int = 1000) -> Tuple[float, float]:
        """Perform statistical comparison with kill line"""
        # Create distribution around prediction
        prediction_dist = np.random.normal(predicted_kills, prediction_std, n_samples)
        
        # Perform t-test against kill line
        t_stat, p_value = stats.ttest_1samp(prediction_dist, kill_line)
        
        # Calculate effect size (Cohen's d)
        effect_size = (predicted_kills - kill_line) / prediction_std
        
        return p_value, effect_size
    
    def predict_matchup(self, matchup: MatchupContext) -> PredictionResult:
        """Predict kills per round for a specific matchup with confidence intervals"""
        print(f"\n=== Predicting for {matchup.player_name} vs {matchup.opponent_team} ===")
        print(f"Tournament: {matchup.tournament} ({matchup.series_type})")
        print(f"Maps: {', '.join(matchup.maps)}")
        print(f"Kill Line: {matchup.kill_line:.3f} kills/round")
        
        # Create features for this matchup
        features = self.create_matchup_features(matchup)
        
        # Predict with uncertainty
        predicted_kills, prediction_std, bootstrap_predictions = self.predict_with_uncertainty(features)
        
        # Calculate confidence intervals
        intervals = self.calculate_confidence_intervals(bootstrap_predictions)
        
        # Statistical comparison with kill line
        p_value, effect_size = self.statistical_comparison(predicted_kills, matchup.kill_line, prediction_std)
        
        # Calculate confidence score based on prediction stability
        confidence_score = max(0, 1 - prediction_std / predicted_kills) if predicted_kills > 0 else 0.5
        
        # Determine recommendation
        if p_value < 0.05 and abs(effect_size) > 0.5:  # Statistically significant and meaningful
            if predicted_kills > matchup.kill_line:
                recommendation = "OVER"
                reasoning = f"Model predicts {predicted_kills:.3f} kills/round vs {matchup.kill_line:.3f} line (p={p_value:.3f}, effect_size={effect_size:.2f})"
            else:
                recommendation = "UNDER"
                reasoning = f"Model predicts {predicted_kills:.3f} kills/round vs {matchup.kill_line:.3f} line (p={p_value:.3f}, effect_size={effect_size:.2f})"
        else:
            recommendation = "UNSURE"
            reasoning = f"Low statistical significance (p={p_value:.3f}) - model predicts {predicted_kills:.3f} kills/round vs {matchup.kill_line:.3f} line"
        
        result = PredictionResult(
            predicted_kills_per_round=predicted_kills,
            confidence_interval_95=intervals['confidence_interval_95'],
            confidence_interval_90=intervals['confidence_interval_90'],
            confidence_interval_80=intervals['confidence_interval_80'],
            prediction_std=prediction_std,
            confidence_score=confidence_score,
            recommendation=recommendation,
            statistical_significance=p_value,
            effect_size=effect_size,
            reasoning=reasoning
        )
        
        # Print results
        print(f"\nPrediction: {predicted_kills:.3f} kills/round (±{prediction_std:.3f})")
        print(f"95% CI: [{intervals['confidence_interval_95'][0]:.3f}, {intervals['confidence_interval_95'][1]:.3f}]")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence_score:.1%}")
        print(f"Statistical significance: p={p_value:.3f}")
        print(f"Effect size: {effect_size:.2f}")
        
        return result
    
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
    """Example usage of the advanced matchup predictor"""
    predictor = AdvancedMatchupPredictor()
    
    if predictor.model is None:
        print("No model available. Please train a model first.")
        return
    
    # Example matchup for a player with many matches
    matchup = MatchupContext(
        player_name="mina",
        player_team="Unknown",
        opponent_team="Unknown",
        tournament="VCT Champions",
        series_type="bo3",
        maps=["Ascent", "Haven"],
        kill_line=28
    )
    
    # Make prediction
    result = predictor.predict_matchup(matchup)
    
    # Save detailed results
    output = {
        'matchup': matchup.to_dict(),
        'prediction': result.to_dict()
    }
    
    with open('matchup_prediction_result.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to matchup_prediction_result.json")

if __name__ == "__main__":
    main() 