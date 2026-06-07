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
from enhanced_data_loader import EnhancedDataLoader, AGENT_ROLES, UNKNOWN_ROLE
from kill_line_fetcher import KillLineFetcher

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
    agent: str = ''   # agent the player is expected to play (optional)
    
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
    def __init__(self, model_path: str = "models/neural_network_gpu_model.pkl",
                 cache_matches: int = 3000):
        """Initialize the advanced predictor with trained model.

        Loads match history once at startup and precomputes per-player/team
        lookup tables so each prediction is a dict lookup rather than a full
        disk scan.  cache_matches controls how many recent match files are
        read for the cache (more = better coverage, slower startup).
        """
        self.load_model(model_path)
        self.data_loader = EnhancedDataLoader()
        self._build_cache(cache_matches)
        self._line_fetcher = KillLineFetcher()

    # ------------------------------------------------------------------
    # One-time cache build
    # ------------------------------------------------------------------

    def _build_cache(self, limit: int):
        """Precompute team strengths and per-player stats from match history."""
        from collections import defaultdict
        print(f"Building prediction cache from {limit} matches...", flush=True)

        matches = self.data_loader.load_scraped_matches(limit=limit)

        team_ratings: dict     = defaultdict(list)
        player_ratings: dict   = defaultdict(list)
        player_kills: dict     = defaultdict(list)
        player_map_kills: dict = defaultdict(list)
        player_agent_kills: dict = defaultdict(list)

        for match in matches:
            for player in match.players:
                if player.rating > 0:
                    team_ratings[player.team].append(player.rating)
                    player_ratings[player.name].append(player.rating)
                if player.kills > 0:
                    player_kills[player.name].append(player.kills)
                    if player.map_name:
                        player_map_kills[(player.name, player.map_name)].append(player.kills)
                    if player.agent:
                        player_agent_kills[(player.name, player.agent.lower())].append(player.kills)

        # Recent form = average of the most recent 10 appearances (list is
        # in file-system order which is roughly chronological by match ID)
        self._cache_team_strength   = {t: float(np.mean(v)) for t, v in team_ratings.items()}
        self._cache_player_rating   = {p: float(np.mean(v[-10:])) for p, v in player_ratings.items()}
        self._cache_player_kills    = {p: float(np.mean(v[-10:])) for p, v in player_kills.items()}
        self._cache_map_kills       = {k: float(np.mean(v)) for k, v in player_map_kills.items()}
        self._cache_agent_kills     = {k: float(np.mean(v)) for k, v in player_agent_kills.items()}

        print(f"Cache ready: {len(self._cache_team_strength)} teams, "
              f"{len(self._cache_player_kills)} players", flush=True)
        
    def load_model(self, model_path: str):
        """Load the trained model — supports both neural network and sklearn models."""
        try:
            self.model_data = joblib.load(model_path)

            if 'model_state_dict' in self.model_data:
                input_size  = self.model_data['input_size']
                hidden_sizes = self.model_data['hidden_sizes']
                self.model = KillPredictionNN(input_size, hidden_sizes=hidden_sizes).to('cpu')
                self.model.load_state_dict(self.model_data['model_state_dict'])
                self.model.eval()
                self.model_type = 'neural_network'
            elif 'model' in self.model_data:
                self.model = self.model_data['model']
                self.model_type = 'sklearn'
            else:
                raise ValueError("Unrecognised model format")

            self.scaler = self.model_data['scaler']
            self.feature_columns = self.model_data['feature_columns']
            perf = self.model_data.get('performance', {})
            print(f"Loaded {self.model_type} model | features={len(self.feature_columns)} | "
                  f"R²={perf.get('r2', '?'):.4f}")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.model_type = None
    
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
        return self._cache_team_strength.get(team_name, 1.0)

    def calculate_map_familiarity(self, player_name: str, maps: List[str]) -> float:
        # Kept for API compatibility; map_avg_kills supersedes this
        vals = [self._cache_map_kills.get((player_name, m)) for m in maps]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals) / 1.5) if vals else 0.5

    def calculate_recent_form(self, player_name: str) -> float:
        return self._cache_player_rating.get(player_name, 1.0)

    def calculate_recent_form_kills(self, player_name: str) -> float:
        return self._cache_player_kills.get(player_name, 13.0)

    def calculate_map_avg_kills(self, player_name: str, maps: List[str]) -> float:
        vals = [self._cache_map_kills.get((player_name, m)) for m in maps]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 13.0

    def calculate_agent_avg_kills(self, player_name: str, agent: str) -> float:
        """Player's historical average kills on this specific agent."""
        if not agent:
            return self._cache_player_kills.get(player_name, 13.0)
        return self._cache_agent_kills.get(
            (player_name, agent.lower()),
            self._cache_player_kills.get(player_name, 13.0),
        )
    
    def fetch_live_kill_line(self, player_name: str) -> Optional[float]:
        """
        Try to fetch a live kill line from PrizePicks.
        Returns the line value, or None if the player has no active projection today.
        """
        line = self._line_fetcher.get_live_line(player_name)
        if line is not None:
            print(f"  [PrizePicks] {player_name}: {line} kills (live)", flush=True)
        return line

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
        """Create feature vector for the specific matchup.

        Returns features in exactly the order stored in self.feature_columns so
        the scaler receives the right values regardless of model version.
        """
        player_stats = self.get_player_stats(matchup.player_name)
        if not player_stats:
            raise ValueError(f"Player {matchup.player_name} not found in database")

        player_team_strength    = self.calculate_team_strength(matchup.player_team)
        opponent_team_strength  = self.calculate_team_strength(matchup.opponent_team)
        recent_avg_kills        = self.calculate_recent_form_kills(matchup.player_name)
        recent_avg_rating       = self.calculate_recent_form(matchup.player_name)
        player_map_avg_kills    = self.calculate_map_avg_kills(matchup.player_name, matchup.maps)
        player_agent_avg_kills  = self.calculate_agent_avg_kills(matchup.player_name, matchup.agent)
        agent_role_ordinal      = float(AGENT_ROLES.get(matchup.agent.lower().strip(), UNKNOWN_ROLE))
        is_duelist              = 1.0 if agent_role_ordinal == 3.0 else 0.0

        features = {
            **player_stats,
            'team_strength':            player_team_strength,
            'opponent_team_strength':   opponent_team_strength,
            'recent_avg_kills':         recent_avg_kills,
            'recent_avg_rating':        recent_avg_rating,
            'player_map_avg_kills':     player_map_avg_kills,
            'agent_role_ordinal':       agent_role_ordinal,
            'is_duelist':               is_duelist,
            'player_agent_avg_kills':   player_agent_avg_kills,
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
            noise = np.random.normal(0, 0.01, len(feature_vector))
            noisy_features = np.array(feature_vector) + noise
            noisy_features_scaled = self.scaler.transform([noisy_features])

            if self.model_type == 'neural_network':
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
        # Auto-fetch live kill line when caller passes 0 / None
        if not matchup.kill_line or matchup.kill_line <= 0:
            live = self.fetch_live_kill_line(matchup.player_name)
            if live is not None:
                matchup = MatchupContext(
                    player_name=matchup.player_name,
                    player_team=matchup.player_team,
                    opponent_team=matchup.opponent_team,
                    tournament=matchup.tournament,
                    series_type=matchup.series_type,
                    maps=matchup.maps,
                    kill_line=live,
                )

        print(f"\n=== Predicting for {matchup.player_name} vs {matchup.opponent_team} ===")
        print(f"Tournament: {matchup.tournament} ({matchup.series_type})")
        print(f"Maps: {', '.join(matchup.maps)}")
        kill_line_label = f"{matchup.kill_line:.1f}" if matchup.kill_line > 0 else "not set"
        print(f"Kill Line: {kill_line_label} kills/map")

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
        print(f"\nPrediction: {predicted_kills:.3f} kills/map (±{prediction_std:.3f})")
        print(f"95% CI: [{intervals['confidence_interval_95'][0]:.3f}, {intervals['confidence_interval_95'][1]:.3f}]")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence_score:.1%}")
        print(f"Statistical significance: p={p_value:.3f}")
        print(f"Effect size: {effect_size:.2f}")
        
        return result
    
    def _get_default_value(self, feature_name: str) -> float:
        """Fallback values for features not found in the player lookup."""
        defaults = {
            'db_rating':                 1.0,
            'db_average_combat_score':   193.0,
            'db_kill_deaths':            0.92,
            'db_kills_per_round':        0.67,
            'db_assists_per_round':      0.27,
            'db_first_kills_per_round':  0.09,
            'db_first_deaths_per_round': 0.10,
            'team_strength':             1.0,
            'opponent_team_strength':    1.0,
            'recent_avg_kills':         13.0,
            'recent_avg_rating':         1.0,
            'player_map_avg_kills':     13.0,
            'agent_role_ordinal':        UNKNOWN_ROLE,
            'is_duelist':                0.0,
            'player_agent_avg_kills':   13.0,
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