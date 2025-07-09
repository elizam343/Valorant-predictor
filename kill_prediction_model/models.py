import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pickle
from enum import Enum

class PredictionType(Enum):
    OVER = 1
    UNDER = 0
    UNSURE = 2

@dataclass
class PredictionResult:
    """Result of a kill line prediction"""
    player_name: str
    team: str
    kill_line: float
    prediction: PredictionType
    confidence: float
    over_probability: float
    under_probability: float
    unsure_probability: float
    recommended_action: str

class KillLinePredictor:
    """Main class for predicting kill line outcomes"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rating', 'average_combat_score', 'kill_deaths', 'kills_per_round',
            'assists_per_round', 'first_kills_per_round', 'first_deaths_per_round',
            'headshot_percentage', 'clutch_success_percentage', 'total_impact',
            'survival_rate', 'efficiency', 'team_avg_rating', 'team_avg_kills',
            'relative_rating', 'relative_kills', 'kill_line'
        ]
        self.confidence_threshold = 0.6  # Minimum confidence to make a prediction
        
    def _create_model(self):
        """Create the specified model type"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train the prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train[self.feature_columns])
        X_test_scaled = self.scaler.transform(X_test[self.feature_columns])
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Under', 'Over', 'Unsure']))
        
        return {
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, player_features: Dict) -> PredictionResult:
        """Make a prediction for a single player"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(player_features.get(col, 0.0))
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Get prediction
        prediction_idx = self.model.predict(feature_vector_scaled)[0]
        prediction = PredictionType(prediction_idx)
        
        # Calculate confidence
        confidence = max(probabilities)
        
        # Determine if we should mark as unsure based on confidence
        if confidence < self.confidence_threshold:
            prediction = PredictionType.UNSURE
        
        # Get recommended action
        if prediction == PredictionType.OVER:
            recommended = "BET OVER"
        elif prediction == PredictionType.UNDER:
            recommended = "BET UNDER"
        else:
            recommended = "AVOID BETTING - Low confidence"
        
        return PredictionResult(
            player_name=player_features.get('player_name', 'Unknown'),
            team=player_features.get('team', 'Unknown'),
            kill_line=player_features.get('kill_line', 0.0),
            prediction=prediction,
            confidence=confidence,
            over_probability=probabilities[1] if len(probabilities) > 1 else 0.0,
            under_probability=probabilities[0] if len(probabilities) > 0 else 0.0,
            unsure_probability=probabilities[2] if len(probabilities) > 2 else 0.0,
            recommended_action=recommended
        )
    
    def predict_batch(self, players_features: List[Dict]) -> List[PredictionResult]:
        """Make predictions for multiple players"""
        return [self.predict(features) for features in players_features]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'confidence_threshold': self.confidence_threshold
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.confidence_threshold = model_data['confidence_threshold']
        print(f"Model loaded from {filepath}")

class EnsemblePredictor:
    """Ensemble model combining multiple prediction approaches"""
    
    def __init__(self, models: List[str] = None):
        if models is None:
            models = ["random_forest", "gradient_boosting", "logistic_regression"]
        
        self.models = {}
        self.weights = {}
        
        for model_type in models:
            self.models[model_type] = KillLinePredictor(model_type)
            self.weights[model_type] = 1.0 / len(models)  # Equal weights initially
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in the ensemble"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            result = model.train(X, y)
            results[name] = result
            
            # Adjust weights based on performance
            self.weights[name] = result['accuracy']
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
        
        print(f"\nEnsemble weights: {self.weights}")
        return results
    
    def predict(self, player_features: Dict) -> PredictionResult:
        """Make ensemble prediction"""
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(player_features)
            predictions.append(pred)
            probabilities.append([
                pred.under_probability,
                pred.over_probability,
                pred.unsure_probability
            ])
        
        # Weighted average of probabilities
        weighted_probs = np.zeros(3)
        for i, (name, weight) in enumerate(self.weights.items()):
            weighted_probs += np.array(probabilities[i]) * weight
        
        # Determine final prediction
        max_prob_idx = np.argmax(weighted_probs)
        confidence = max(weighted_probs)
        
        if confidence < 0.6:  # Confidence threshold
            prediction = PredictionType.UNSURE
        else:
            prediction = PredictionType(max_prob_idx)
        
        # Get recommended action
        if prediction == PredictionType.OVER:
            recommended = "BET OVER"
        elif prediction == PredictionType.UNDER:
            recommended = "BET UNDER"
        else:
            recommended = "AVOID BETTING - Low confidence"
        
        return PredictionResult(
            player_name=player_features.get('player_name', 'Unknown'),
            team=player_features.get('team', 'Unknown'),
            kill_line=player_features.get('kill_line', 0.0),
            prediction=prediction,
            confidence=confidence,
            over_probability=weighted_probs[1],
            under_probability=weighted_probs[0],
            unsure_probability=weighted_probs[2],
            recommended_action=recommended
        )
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble model"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble model saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load the ensemble model"""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        print(f"Ensemble model loaded from {filepath}") 