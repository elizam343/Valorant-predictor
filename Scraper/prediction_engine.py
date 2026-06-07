#!/usr/bin/env python3
"""
Prediction engine for Valorant performance predictions.
Wraps AdvancedMatchupPredictor for use by the Flask app.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kill_prediction_model'))

from db_utils import get_players, get_teams
from advanced_matchup_predictor import AdvancedMatchupPredictor, MatchupContext


class PredictionEngine:
    def __init__(self):
        self._players_cache = None
        self._teams_cache = None

        # Try gradient boosting first (usually better), fall back to neural net
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'kill_prediction_model', 'models')
        for model_file in ('gradient_boosting_gpu_model.pkl', 'neural_network_gpu_model.pkl'):
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    self.predictor = AdvancedMatchupPredictor(model_path=model_path)
                    if self.predictor.model is not None:
                        print(f"PredictionEngine loaded: {model_file}")
                        return
                except Exception as e:
                    print(f"Could not load {model_file}: {e}")

        print("Warning: no model loaded — predictions will be unavailable")
        self.predictor = None

    # ------------------------------------------------------------------
    # Player / team helpers
    # ------------------------------------------------------------------

    def get_available_players(self):
        if self._players_cache is None:
            rows = get_players()
            self._players_cache = [
                {
                    'name': r[1],
                    'team': r[2],
                    'rating': r[3],
                    'kills_per_round': r[8],
                    'average_combat_score': r[4],
                }
                for r in rows
                if r[3] and float(r[3] or 0) > 0
            ]
        return self._players_cache

    def get_available_teams(self):
        if self._teams_cache is None:
            teams = get_teams()
            self._teams_cache = [
                t for t in teams
                if t and not t.startswith('(+') and not str(t).isdigit()
            ]
        return sorted(self._teams_cache)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_performance(self, player_name, player_team, opponent_team,
                            map_name, series_type, kill_line=None,
                            tournament='Unknown Tournament'):
        if self.predictor is None:
            return {'error': 'Model not loaded'}

        try:
            maps = [map_name] if map_name else ['Ascent']
            kl = float(kill_line) if kill_line is not None else 13.0

            matchup = MatchupContext(
                player_name=player_name,
                player_team=player_team,
                opponent_team=opponent_team,
                tournament=tournament,
                series_type=series_type.lower(),
                maps=maps,
                kill_line=kl,
            )

            result = self.predictor.predict_matchup(matchup)

            return {
                'player_name': player_name,
                'predicted_kills': round(result.predicted_kills_per_round, 2),
                'kill_line': kl,
                'recommendation': result.recommendation,
                'confidence': round(result.confidence_score, 4),
                'confidence_interval_95': [
                    round(result.confidence_interval_95[0], 2),
                    round(result.confidence_interval_95[1], 2),
                ],
                'p_value': round(float(result.statistical_significance), 4),
                'effect_size': round(float(result.effect_size), 3),
                'reasoning': result.reasoning,
            }

        except Exception as e:
            return {'error': str(e)}

    def get_performance_insights(self, player_name, team_name):
        players = self.get_available_players()
        player = next(
            (p for p in players if p['name'] == player_name and p['team'] == team_name),
            None
        )
        if not player:
            return None

        rating = float(player.get('rating') or 0)
        kpr    = float(player.get('kills_per_round') or 0)
        acs    = float(player.get('average_combat_score') or 0)

        strengths, weaknesses, recommendations = [], [], []

        if rating > 1.2:
            strengths.append('High overall rating')
        if kpr > 0.8:
            strengths.append('Strong kill performance')
        if acs > 250:
            strengths.append('High average combat score')
        if rating < 0.9:
            weaknesses.append('Below-average rating')
        if kpr < 0.6:
            weaknesses.append('Low kills per round')
        if kpr < 0.7:
            recommendations.append('Focus on aggressive positioning')

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
        }
