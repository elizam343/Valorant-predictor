import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import Dict, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta

class ValorantKillPredictor:
    """Production-ready Valorant kill prediction service"""
    
    def __init__(self, model_path: str = None, db_path: str = None):
        """Initialize the prediction service"""
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_performance = {}
        self.db_path = db_path or "../Scraper/valorant_matches.db"
        
        # Try to load the best available model
        model_paths = [
            model_path,
            "models/precision_model.pkl",
            "models/stable_model.pkl", 
            "models/gpu_maximized_model.pkl",
            "../google_colab/precision_model.pkl",
            "../google_colab/stable_model.pkl",
            "../google_colab/gpu_maximized_model.pkl"
        ]
        
        for path in model_paths:
            if path and os.path.exists(path):
                self.load_model(path)
                break
        
        if self.model is None:
            print("⚠️ No trained model found. Please ensure model files are available.")
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained model from file"""
        try:
            print(f"🔄 Loading model from: {model_path}")
            
            # Handle CUDA models on CPU-only machines
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_package = torch.load(model_path, map_location=device)
            
            # Extract model components
            self.model_state = model_package['model_state_dict']
            self.feature_names = model_package['feature_names']
            self.scaler = model_package['scaler']
            self.model_performance = model_package.get('performance', {})
            
            # Determine model architecture
            model_type = model_package.get('model_type', 'stable_optimized_v1')
            input_size = len(self.feature_names)
            
            # Recreate the appropriate model architecture
            if 'precision' in model_type.lower():
                print(f"   🎯 Using Precision-Tuned architecture")
                from neural_models import PrecisionTunedNN
                self.model = PrecisionTunedNN(input_size)
            else:
                print(f"   🛡️ Using Stable architecture")
                from neural_models import StableKillPredictionNN
                self.model = StableKillPredictionNN(input_size)
            
            # Load the state dict and set to evaluation mode
            self.model.load_state_dict(self.model_state)
            self.model.eval()
            
            # Move model to appropriate device
            self.model = self.model.to(device)
            
            print(f"✅ Model loaded successfully on {device}")
            print(f"📊 Features: {len(self.feature_names)}")
            print(f"🎯 Performance: MAE = {self.model_performance.get('mae', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_player_features(self, player_name: str, opponent_team: str = None, 
                           map_name: str = None, series_type: str = "bo3", 
                           tournament: str = "regional") -> Optional[Dict]:
        """Extract features for a player from the database"""
        try:
            if not os.path.exists(self.db_path):
                print(f"⚠️ Database not found at: {self.db_path}")
                return None
            
            conn = sqlite3.connect(self.db_path)
            
            # Query for player data
            query = """
            SELECT 
                p.name as player_name, t.name as team_name,
                m.match_date, m.series_type, tour.name as tournament_name,
                mp.map_name, pms.kills, pms.deaths, pms.assists, 
                pms.acs, pms.adr, pms.fk, pms.hs_percentage, pms.kdr,
                m.match_id, pms.map_id
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.id
            JOIN teams t ON pms.team_id = t.id
            JOIN matches m ON pms.match_id = m.id
            JOIN maps mp ON pms.map_id = mp.id
            JOIN tournaments tour ON m.tournament_id = tour.id
            WHERE p.name LIKE ?
            ORDER BY m.match_date DESC, pms.map_id
            LIMIT 50
            """
            
            df = pd.read_sql_query(query, conn, params=[f"%{player_name}%"])
            conn.close()
            
            if len(df) == 0:
                return None
            
            # Use most recent player name match
            actual_player_name = df.iloc[0]['player_name']
            
            # Calculate features (simplified version of the training features)
            features = self._calculate_player_features(df, actual_player_name, opponent_team, 
                                                     map_name, series_type, tournament)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting player features: {e}")
            return None
    
    def _calculate_player_features(self, df: pd.DataFrame, player_name: str,
                                 opponent_team: str, map_name: str, 
                                 series_type: str, tournament: str) -> Dict:
        """Calculate the same features used in training"""
        
        # Basic historical features
        recent_games = df.head(15)  # Last 15 games
        
        features = {
            'hist_avg_kills': recent_games['kills'].mean(),
            'hist_avg_kdr': recent_games['kdr'].mean(),
            'recent_3_avg': df.head(3)['kills'].mean() if len(df) >= 3 else df['kills'].mean(),
            'recent_5_avg': df.head(5)['kills'].mean() if len(df) >= 5 else df['kills'].mean(),
            'recent_10_avg': df.head(10)['kills'].mean() if len(df) >= 10 else df['kills'].mean(),
        }
        
        # Momentum and consistency
        if len(df) >= 3:
            features['momentum_trend'] = features['recent_3_avg'] - features['recent_10_avg']
            features['form_acceleration'] = features['recent_3_avg'] - features['recent_5_avg']
        else:
            features['momentum_trend'] = 0.0
            features['form_acceleration'] = 0.0
        
        # Kill consistency (inverse of standard deviation)
        kill_std = recent_games['kills'].std() if len(recent_games) > 1 else 1.0
        features['kill_consistency'] = 1 / (1 + kill_std)
        
        features['performance_vs_expectation'] = features['recent_5_avg'] - features['hist_avg_kills']
        
        # Time-based features
        if len(df) >= 2:
            df['match_date'] = pd.to_datetime(df['match_date'])
            last_match_days = (datetime.now() - df.iloc[0]['match_date']).days
            features['days_since_last'] = min(30, max(0, last_match_days))
        else:
            features['days_since_last'] = 7.0
        
        # Rest factor
        if features['days_since_last'] <= 1:
            features['rest_factor'] = 1.0
        elif features['days_since_last'] <= 7:
            features['rest_factor'] = 1.05
        elif features['days_since_last'] <= 14:
            features['rest_factor'] = 0.98
        else:
            features['rest_factor'] = 0.95
        
        # Tournament and match context
        tournament_tiers = {
            'champions': 1.0, 'masters': 0.95, 'regional': 0.85, 
            'qualifier': 0.75, 'other': 0.70
        }
        features['tournament_tier_weight'] = tournament_tiers.get(
            tournament.lower(), 0.75
        )
        
        series_importance = {'bo1': 1.2, 'bo3': 1.0, 'bo5': 0.9}
        features['series_pressure'] = series_importance.get(series_type.lower(), 1.0)
        features['match_importance'] = features['tournament_tier_weight'] * features['series_pressure']
        
        # Map and team features (defaults)
        features['agent_kill_expectation'] = 1.0
        features['map_specialization'] = 1.0
        features['map_kill_factor'] = 1.0
        features['team_synergy_factor'] = 1.0
        features['estimated_rounds'] = 22.0  # Average rounds per map
        features['game_competitiveness'] = 0.5
        
        # Player role (simplified classification)
        avg_kills = features['hist_avg_kills']
        if avg_kills >= 18:
            role_expectation = 1.25  # Star fragger
            role_encoded = 4
        elif avg_kills >= 15:
            role_expectation = 1.10  # Secondary fragger  
            role_encoded = 3
        elif avg_kills >= 12:
            role_expectation = 1.00  # Balanced player
            role_encoded = 2
        else:
            role_expectation = 0.85  # Support player
            role_encoded = 1
        
        features['role_kill_expectation'] = role_expectation
        features['player_role_encoded'] = role_encoded
        features['series_type_encoded'] = {'bo1': 0, 'bo3': 1, 'bo5': 2}.get(series_type.lower(), 1)
        
        # H2H and interaction features (defaults)
        features['h2h_avg_kills'] = features['hist_avg_kills']
        features['h2h_trend'] = 0.0
        features['h2h_consistency'] = 0.5
        features['h2h_experience'] = min(len(df), 10)
        
        # Confidence weight based on experience
        total_maps = len(df)
        if total_maps >= 30:
            features['confidence_weight'] = 1.0
        elif total_maps >= 20:
            features['confidence_weight'] = 0.95
        elif total_maps >= 10:
            features['confidence_weight'] = 0.85
        else:
            features['confidence_weight'] = 0.70
        
        # Interaction features
        features['role_map_interaction'] = features['role_kill_expectation'] * features['map_specialization']
        features['form_importance_interaction'] = features['momentum_trend'] * features['match_importance']
        features['experience_confidence'] = features['h2h_experience'] * features['confidence_weight']
        features['consistency_expectation'] = features['kill_consistency'] * features['role_kill_expectation']
        
        return features
    
    def predict_kills(self, player_name: str, opponent_team: str = None,
                     map_name: str = None, series_type: str = "bo3",
                     tournament: str = "regional", kill_line: float = None) -> Dict:
        """Make a kill prediction for a player"""
        
        if self.model is None:
            return {
                'success': False,
                'error': 'No model loaded',
                'player': player_name
            }
        
        # Get player features
        features = self.get_player_features(player_name, opponent_team, map_name, 
                                          series_type, tournament)
        
        if features is None:
            return {
                'success': False,
                'error': f'Player "{player_name}" not found in database',
                'suggestion': 'Check spelling or try partial name',
                'player': player_name
            }
        
        try:
            # Prepare features for prediction
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale and predict
            features_scaled = self.scaler.transform([feature_vector])
            features_tensor = torch.FloatTensor(features_scaled)
            
            with torch.no_grad():
                raw_prediction = self.model(features_tensor).numpy()
                if raw_prediction.ndim == 0:
                    predicted_kills = float(raw_prediction)
                else:
                    predicted_kills = float(raw_prediction[0])
                
                predicted_kills = np.clip(predicted_kills, 0, 35)
            
            # Calculate confidence based on player experience and model uncertainty
            confidence = self._calculate_confidence(features, predicted_kills)
            
            # Generate recommendation if kill line provided
            recommendation = None
            if kill_line is not None:
                recommendation = self._generate_recommendation(predicted_kills, kill_line, confidence)
            
            return {
                'success': True,
                'player': player_name,
                'predicted_kills': round(predicted_kills, 1),
                'confidence': round(confidence * 100, 1),
                'confidence_level': self._get_confidence_level(confidence),
                'player_stats': {
                    'recent_avg': round(features['recent_5_avg'], 1),
                    'historical_avg': round(features['hist_avg_kills'], 1),
                    'momentum': round(features['momentum_trend'], 2),
                    'experience_maps': int(features.get('total_maps', features['h2h_experience'])),
                    'role': self._get_player_role(features['role_kill_expectation'])
                },
                'match_context': {
                    'opponent': opponent_team,
                    'map': map_name,
                    'series_type': series_type.upper(),
                    'tournament': tournament
                },
                'recommendation': recommendation,
                'model_performance': {
                    'mae': self.model_performance.get('mae', 'Unknown'),
                    'accuracy_note': f"Typically accurate within ±{self.model_performance.get('mae', 3.3):.1f} kills"
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'player': player_name
            }
    
    def _calculate_confidence(self, features: Dict, predicted_kills: float) -> float:
        """Calculate prediction confidence based on data quality and player consistency"""
        
        confidence_factors = []
        
        # Experience factor (more games = higher confidence)
        experience = features.get('h2h_experience', 5)
        experience_confidence = min(1.0, experience / 20)  # Max confidence at 20+ games
        confidence_factors.append(experience_confidence)
        
        # Consistency factor (consistent players = higher confidence)
        consistency = features.get('kill_consistency', 0.5)
        confidence_factors.append(consistency)
        
        # Recent activity factor (recent games = higher confidence)
        days_since = features.get('days_since_last', 7)
        recency_confidence = max(0.5, 1.0 - (days_since / 30))  # Decay over 30 days
        confidence_factors.append(recency_confidence)
        
        # Model confidence factor (based on experience with this prediction range)
        if 10 <= predicted_kills <= 25:  # Common range
            range_confidence = 0.9
        elif 5 <= predicted_kills <= 30:  # Extended range
            range_confidence = 0.8
        else:  # Extreme predictions
            range_confidence = 0.6
        confidence_factors.append(range_confidence)
        
        # Calculate weighted average
        overall_confidence = np.mean(confidence_factors)
        
        # Apply experience weight
        experience_weight = features.get('confidence_weight', 0.8)
        final_confidence = overall_confidence * experience_weight
        
        return min(0.95, max(0.3, final_confidence))  # Bound between 30-95%
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to readable level"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _get_player_role(self, role_expectation: float) -> str:
        """Convert role expectation to readable role"""
        if role_expectation >= 1.20:
            return "Star Fragger"
        elif role_expectation >= 1.05:
            return "Secondary Fragger"
        elif role_expectation >= 0.95:
            return "Balanced Player"
        else:
            return "Support Player"
    
    def _generate_recommendation(self, predicted_kills: float, kill_line: float, confidence: float) -> Dict:
        """Generate betting recommendation"""
        difference = predicted_kills - kill_line
        percentage_diff = (difference / kill_line) * 100 if kill_line > 0 else 0
        
        # Confidence threshold for recommendations
        min_confidence = 0.65  # 65% minimum confidence
        min_difference = 1.5   # Must be off by at least 1.5 kills
        
        if confidence >= min_confidence and abs(difference) >= min_difference:
            if difference > 0:
                action = "OVER"
                reasoning = f"Model predicts {predicted_kills:.1f} kills vs {kill_line:.1f} line (+{difference:.1f})"
            else:
                action = "UNDER"  
                reasoning = f"Model predicts {predicted_kills:.1f} kills vs {kill_line:.1f} line ({difference:.1f})"
        else:
            action = "UNSURE"
            if confidence < min_confidence:
                reasoning = f"Low confidence ({confidence*100:.0f}%) - avoid betting"
            else:
                reasoning = f"Close to line ({difference:+.1f} kills) - avoid betting"
        
        return {
            'action': action,
            'reasoning': reasoning,
            'difference': round(difference, 1),
            'percentage_diff': round(percentage_diff, 1),
            'kill_line': kill_line,
            'confidence_met': confidence >= min_confidence,
            'difference_met': abs(difference) >= min_difference
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'performance': self.model_performance,
            'features_count': len(self.feature_names),
            'database_path': self.db_path,
            'database_available': os.path.exists(self.db_path)
        } 

    def predict_series_kills(self, player_name: str, opponent_team: str = None,
                           maps: list = None, series_type: str = "bo3", 
                           tournament: str = "regional", kill_line: float = None,
                           maps_to_predict: int = 2) -> Dict:
        """Make cumulative kill predictions for multiple maps in a series"""
        
        if self.model is None:
            return {
                'success': False,
                'error': 'No model loaded',
                'player': player_name
            }
        
        # Validate maps_to_predict
        if maps_to_predict not in [2, 3]:
            return {
                'success': False,
                'error': 'maps_to_predict must be 2 or 3',
                'player': player_name
            }
        
        # Get player features
        features = self.get_player_features(player_name, opponent_team, None, 
                                          series_type, tournament)
        
        if features is None:
            return {
                'success': False,
                'error': f'Player "{player_name}" not found in database',
                'suggestion': 'Check spelling or try partial name',
                'player': player_name
            }
        
        try:
            # If specific maps provided, predict each map individually
            if maps and len(maps) >= maps_to_predict:
                map_predictions = []
                total_predicted_kills = 0
                
                for i in range(maps_to_predict):
                    map_name = maps[i]
                    
                    # Get map-specific features (update map context)
                    map_features = self.get_player_features(player_name, opponent_team, 
                                                          map_name, series_type, tournament)
                    if map_features:
                        features_for_map = map_features
                    else:
                        features_for_map = features  # Fallback to general features
                    
                    # Make individual map prediction
                    map_prediction = self._predict_single_map(features_for_map)
                    map_predictions.append({
                        'map': map_name,
                        'predicted_kills': map_prediction,
                        'map_order': i + 1
                    })
                    total_predicted_kills += map_prediction
                
            else:
                # Predict using average map performance
                single_map_prediction = self._predict_single_map(features)
                
                # Adjust for series context and map count
                series_adjustment = self._calculate_series_adjustment(features, maps_to_predict, series_type)
                total_predicted_kills = single_map_prediction * maps_to_predict * series_adjustment
                
                # Create placeholder map predictions
                map_predictions = []
                avg_per_map = total_predicted_kills / maps_to_predict
                for i in range(maps_to_predict):
                    map_predictions.append({
                        'map': maps[i] if maps and len(maps) > i else f'Map {i+1}',
                        'predicted_kills': avg_per_map,
                        'map_order': i + 1
                    })
            
            # Calculate series-level confidence
            base_confidence = self._calculate_confidence(features, total_predicted_kills / maps_to_predict)
            # Reduce confidence slightly for multi-map predictions due to increased uncertainty
            series_confidence = base_confidence * (0.95 if maps_to_predict == 2 else 0.90)
            
            # Generate recommendation if kill line provided
            recommendation = None
            if kill_line is not None:
                recommendation = self._generate_series_recommendation(
                    total_predicted_kills, kill_line, series_confidence, maps_to_predict
                )
            
            return {
                'success': True,
                'player': player_name,
                'series_type': f'First {maps_to_predict} maps',
                'predicted_kills': round(total_predicted_kills, 1),
                'average_per_map': round(total_predicted_kills / maps_to_predict, 1),
                'confidence': round(series_confidence * 100, 1),
                'confidence_level': self._get_confidence_level(series_confidence),
                'map_predictions': map_predictions,
                'player_stats': {
                    'recent_avg': round(features['recent_5_avg'], 1),
                    'historical_avg': round(features['hist_avg_kills'], 1),
                    'momentum': round(features['momentum_trend'], 2),
                    'experience_maps': int(features.get('total_maps', features['h2h_experience'])),
                    'role': self._get_player_role(features['role_kill_expectation'])
                },
                'series_context': {
                    'opponent': opponent_team,
                    'maps': maps[:maps_to_predict] if maps else [f'Map {i+1}' for i in range(maps_to_predict)],
                    'series_type': series_type.upper(),
                    'tournament': tournament,
                    'maps_count': maps_to_predict
                },
                'recommendation': recommendation,
                'model_performance': {
                    'mae': self.model_performance.get('mae', 'Unknown'),
                    'accuracy_note': f"Typically accurate within ±{self.model_performance.get('mae', 3.3) * maps_to_predict:.1f} kills for {maps_to_predict} maps"
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Series prediction failed: {str(e)}',
                'player': player_name
            }
    
    def _predict_single_map(self, features: Dict) -> float:
        """Make a single map prediction using the model"""
        # Prepare features for prediction
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0.0))
        
        # Scale and predict
        features_scaled = self.scaler.transform([feature_vector])
        features_tensor = torch.FloatTensor(features_scaled)
        
        with torch.no_grad():
            raw_prediction = self.model(features_tensor).numpy()
            if raw_prediction.ndim == 0:
                predicted_kills = float(raw_prediction)
            else:
                predicted_kills = float(raw_prediction[0])
            
            predicted_kills = np.clip(predicted_kills, 0, 35)
        
        return predicted_kills
    
    def _calculate_series_adjustment(self, features: Dict, maps_count: int, series_type: str) -> float:
        """Calculate adjustment factor for multi-map predictions"""
        base_adjustment = 1.0
        
        # Player consistency affects series performance
        consistency = features.get('kill_consistency', 0.5)
        if consistency > 0.7:  # Consistent players perform similarly across maps
            adjustment = 1.0
        elif consistency < 0.4:  # Inconsistent players have more variance
            adjustment = 0.95  # Slightly conservative
        else:
            adjustment = 0.98
        
        # Series type adjustment
        if series_type.lower() == 'bo5':
            # BO5 series tend to be longer, potential fatigue
            adjustment *= 0.98
        elif series_type.lower() == 'bo1':
            # Single map, players might play more aggressively
            adjustment *= 1.02
        
        # Tournament pressure
        tournament_weight = features.get('tournament_tier_weight', 0.75)
        if tournament_weight >= 0.95:  # High stakes
            adjustment *= 0.97  # Slightly more conservative
        
        return adjustment
    
    def _generate_series_recommendation(self, predicted_kills: float, kill_line: float, 
                                      confidence: float, maps_count: int) -> Dict:
        """Generate betting recommendation for series predictions"""
        difference = predicted_kills - kill_line
        percentage_diff = (difference / kill_line) * 100 if kill_line > 0 else 0
        
        # Adjusted thresholds for series predictions
        min_confidence = 0.60  # Slightly lower for series (more uncertainty)
        min_difference = 2.0 if maps_count == 2 else 3.0  # Higher threshold for more maps
        
        if confidence >= min_confidence and abs(difference) >= min_difference:
            if difference > 0:
                action = "OVER"
                reasoning = f"Model predicts {predicted_kills:.1f} kills vs {kill_line:.1f} line across {maps_count} maps (+{difference:.1f})"
            else:
                action = "UNDER"  
                reasoning = f"Model predicts {predicted_kills:.1f} kills vs {kill_line:.1f} line across {maps_count} maps ({difference:.1f})"
        else:
            action = "UNSURE"
            if confidence < min_confidence:
                reasoning = f"Low confidence ({confidence*100:.0f}%) for {maps_count}-map series - avoid betting"
            else:
                reasoning = f"Close to line ({difference:+.1f} kills) across {maps_count} maps - avoid betting"
        
        return {
            'action': action,
            'reasoning': reasoning,
            'difference': round(difference, 1),
            'percentage_diff': round(percentage_diff, 1),
            'kill_line': kill_line,
            'confidence_met': confidence >= min_confidence,
            'difference_met': abs(difference) >= min_difference,
            'maps_count': maps_count
        } 