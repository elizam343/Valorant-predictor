#!/usr/bin/env python3
"""
Prediction engine for Valorant performance predictions
Integrates with the trained performance model
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add the kill_prediction_model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kill_prediction_model'))

from data_loader import DataLoader

class PredictionEngine:
    """Engine for making performance predictions"""
    
    def __init__(self, model_path="../kill_prediction_model/models/performance_predictor_2000.pkl"):
        self.model = None
        self.feature_columns = None
        self.analyzer = None
        self.data_loader = DataLoader()
        
        # Try to load the model
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.analyzer = model_data['analyzer']
            print(f"Loaded performance model with accuracy: {model_data['accuracy']:.3f}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Prediction features will be limited")
    
    def get_available_players(self):
        """Get list of available players for prediction"""
        try:
            players_df = self.data_loader.load_all_players()
            players_df = self.data_loader.create_features(players_df)
            
            # Filter for quality players
            players_df = players_df.dropna(subset=['kills_per_round', 'rating', 'average_combat_score'])
            players_df = players_df[players_df['kills_per_round'] > 0.3]
            players_df = players_df[players_df['rating'] > 0.8]
            
            return players_df[['name', 'team', 'rating', 'kills_per_round', 'average_combat_score']].to_dict('records')
        except Exception as e:
            print(f"Error loading players: {e}")
            return []
    
    def get_available_teams(self):
        """Get list of available teams"""
        try:
            players_df = self.data_loader.load_all_players()
            teams = players_df['team'].unique().tolist()
            # Filter out placeholder teams
            real_teams = [team for team in teams if not team.startswith('(+') and not team.isdigit()]
            return sorted(real_teams)
        except Exception as e:
            print(f"Error loading teams: {e}")
            return []
    
    def predict_performance(self, player_name, player_team, opponent_team, map_name, series_type):
        """
        Predict player performance in a specific scenario
        Returns prediction details and confidence
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0,
                'details': {}
            }
        
        try:
            # Get player data
            players_df = self.data_loader.load_all_players()
            players_df = self.data_loader.create_features(players_df)
            
            # Find the player
            player_data = players_df[
                (players_df['name'] == player_name) & 
                (players_df['team'] == player_team)
            ]
            
            if player_data.empty:
                return {
                    'error': f'Player {player_name} not found in team {player_team}',
                    'prediction': None,
                    'confidence': 0,
                    'details': {}
                }
            
            player_row = player_data.iloc[0]
            
            # Create prediction scenario
            scenario = self._create_prediction_scenario(
                player_row, opponent_team, map_name, series_type
            )
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(scenario)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            prediction_proba = self.model.predict_proba([features])[0]
            
            # Get confidence
            confidence = max(prediction_proba)
            
            # Create detailed response
            result = {
                'prediction': int(prediction),
                'prediction_label': self._get_prediction_label(prediction),
                'confidence': float(confidence),
                'probability_distribution': {
                    'under_perform': float(prediction_proba[0]),
                    'meet_expectations': float(prediction_proba[2]),
                    'over_perform': float(prediction_proba[1])
                },
                'player_stats': {
                    'name': player_name,
                    'team': player_team,
                    'rating': float(player_row['rating']),
                    'kills_per_round': float(player_row['kills_per_round']),
                    'average_combat_score': float(player_row['average_combat_score']),
                    'kill_deaths': float(player_row['kill_deaths']),
                    'headshot_percentage': float(player_row['headshot_percentage'])
                },
                'scenario': {
                    'opponent_team': opponent_team,
                    'map': map_name,
                    'series_type': series_type,
                    'expected_kills': float(scenario['expected_kills']),
                    'performance_consistency': float(scenario['performance_consistency'])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': 0,
                'details': {}
            }
    
    def _create_prediction_scenario(self, player_row, opponent_team, map_name, series_type):
        """Create a prediction scenario"""
        # Determine games to analyze based on series type
        if series_type == "BO1":
            games_to_analyze = 1
        elif series_type == "BO3":
            games_to_analyze = 2
        else:  # BO5
            games_to_analyze = 3
        
        # Calculate expected performance
        expected_kills = self._calculate_expected_performance(
            player_row, opponent_team, map_name, 0, series_type
        )
        
        # Calculate performance consistency (simplified)
        performance_consistency = 1.0 / (1.0 + player_row.get('rating_std', 0.1))
        
        return {
            'player_row': player_row,
            'expected_kills': expected_kills,
            'performance_consistency': performance_consistency,
            'games_to_analyze': games_to_analyze,
            'series_type': series_type,
            'map_name': map_name
        }
    
    def _calculate_expected_performance(self, player, opponent_team, map_name, game_num, series_type):
        """Calculate expected kills based on context"""
        base_kills = player['kills_per_round']
        
        # Map adjustment
        map_adjustment = self._get_map_adjustment(map_name)
        
        # Series progression adjustment
        series_adjustment = self._get_series_adjustment(game_num, series_type)
        
        # Opponent adjustment (simplified)
        opponent_adjustment = 1.0  # Could be enhanced with opponent analysis
        
        # Player form
        form_factor = player['rating']
        
        expected_kills = base_kills * map_adjustment * series_adjustment * opponent_adjustment * form_factor
        return max(0.5, expected_kills)
    
    def _get_map_adjustment(self, map_name):
        """Get map-specific performance adjustment"""
        map_adjustments = {
            'Ascent': 1.0,
            'Haven': 1.05,
            'Split': 0.95,
            'Bind': 0.9,
            'Icebox': 1.1,
            'Breeze': 1.0,
            'Fracture': 0.95,
            'Pearl': 1.0,
            'Lotus': 1.05,
            'Sunset': 1.0
        }
        return map_adjustments.get(map_name, 1.0)
    
    def _get_series_adjustment(self, game_num, series_type):
        """Get series progression adjustment"""
        if series_type == "BO1":
            return 1.0
        elif game_num == 0:
            return 1.0
        elif game_num == 1:
            return 1.0
        else:
            return 0.95
    
    def _prepare_prediction_features(self, scenario):
        """Prepare feature vector for prediction"""
        # Create a proper feature vector that matches our training data
        features = [0.0] * 27  # Match the number of features in our model
        
        # Get player data for the scenario
        player_row = scenario.get('player_row')
        if player_row is not None:
            # Player base stats
            features[0] = float(player_row['rating'])  # rating
            features[1] = float(player_row['average_combat_score'])  # average_combat_score
            features[2] = float(player_row['kill_deaths'])  # kill_deaths
            features[3] = float(player_row['kills_per_round'])  # kills_per_round
            features[4] = float(player_row['assists_per_round'])  # assists_per_round
            features[5] = float(player_row['first_kills_per_round'])  # first_kills_per_round
            features[6] = float(player_row['first_deaths_per_round'])  # first_deaths_per_round
            features[7] = float(player_row['headshot_percentage'])  # headshot_percentage
            features[8] = float(player_row['clutch_success_percentage'])  # clutch_success_percentage
            
            # Derived features (simplified calculations)
            features[9] = float(player_row['kills_per_round'] + player_row['assists_per_round'])  # total_impact
            features[10] = float(1 - player_row['first_deaths_per_round'])  # survival_rate
            features[11] = float(player_row['kills_per_round'] / (player_row['kills_per_round'] + player_row['kills_per_round'] / player_row['kill_deaths']).replace(0, 1))  # efficiency
            features[12] = float(player_row['kills_per_round'] / player_row['average_combat_score'].replace(0, 1))  # kills_per_acs
            features[13] = float(player_row['average_damage_per_round'] / player_row['kills_per_round'].replace(0, 1))  # damage_per_kill
            features[14] = float(player_row['clutch_success_percentage'] * player_row['kills_per_round'])  # clutch_impact
            features[15] = float(player_row['headshot_percentage'] * player_row['kills_per_round'])  # headshot_efficiency
            
            # Team relative features (simplified)
            features[16] = 0.0  # relative_rating
            features[17] = 0.0  # relative_kills
            features[18] = 0.0  # relative_acs
            features[19] = 1.0  # consistency_score
        
        # Context features
        features[20] = self._encode_series_type(scenario.get('series_type', 'BO3'))  # series_type
        features[21] = float(scenario.get('games_to_analyze', 2))  # games_to_analyze
        features[22] = self._get_map_factor(scenario.get('map_name', 'Ascent'))  # map_factor
        
        # Performance expectation features
        features[23] = float(scenario.get('expected_kills', 15.0))  # expected_kills
        features[24] = float(scenario.get('performance_consistency', 0.8))  # performance_consistency
        features[25] = 0.0  # headshot_advantage
        features[26] = 0.0  # clutch_advantage
        
        return features
    
    def _get_prediction_label(self, prediction):
        """Convert prediction number to label"""
        labels = {
            0: "Under-perform",
            1: "Over-perform", 
            2: "Meet expectations"
        }
        return labels.get(prediction, "Unknown")
    
    def _encode_series_type(self, series_type):
        """Encode series type as numeric value"""
        encoding = {
            'BO1': 0.3,
            'BO3': 0.7,
            'BO5': 1.0
        }
        return encoding.get(series_type, 0.5)
    
    def _get_map_factor(self, map_name):
        """Get map-specific factor"""
        map_factors = {
            'Ascent': 1.0,
            'Haven': 1.05,
            'Split': 0.95,
            'Bind': 0.9,
            'Icebox': 1.1,
            'Breeze': 1.0,
            'Fracture': 0.95,
            'Pearl': 1.0,
            'Lotus': 1.05,
            'Sunset': 1.0
        }
        return map_factors.get(map_name, 1.0)
    
    def get_prediction_history(self):
        """Get recent prediction history (placeholder)"""
        return []
    
    def get_performance_insights(self, player_name, team_name):
        """Get performance insights for a player"""
        try:
            players_df = self.data_loader.load_all_players()
            players_df = self.data_loader.create_features(players_df)
            
            player_data = players_df[
                (players_df['name'] == player_name) & 
                (players_df['team'] == team_name)
            ]
            
            if player_data.empty:
                return None
            
            player = player_data.iloc[0]
            
            insights = {
                'strengths': [],
                'weaknesses': [],
                'recommendations': []
            }
            
            # Analyze strengths
            if player['rating'] > 1.2:
                insights['strengths'].append("High overall rating")
            if player['kills_per_round'] > 0.8:
                insights['strengths'].append("Strong kill performance")
            if player['headshot_percentage'] > 60:
                insights['strengths'].append("Excellent aim accuracy")
            if player['clutch_success_percentage'] > 50:
                insights['strengths'].append("Good clutch performance")
            
            # Analyze weaknesses
            if player['rating'] < 0.9:
                insights['weaknesses'].append("Below average rating")
            if player['kill_deaths'] < 0.8:
                insights['weaknesses'].append("Low kill/death ratio")
            if player['first_deaths_per_round'] > 0.3:
                insights['weaknesses'].append("High first death rate")
            
            # Recommendations
            if player['headshot_percentage'] < 40:
                insights['recommendations'].append("Focus on aim training")
            if player['clutch_success_percentage'] < 30:
                insights['recommendations'].append("Improve clutch situations")
            if player['first_kills_per_round'] < 0.1:
                insights['recommendations'].append("Work on entry fragging")
            
            return insights
            
        except Exception as e:
            print(f"Error getting insights: {e}")
            return None 