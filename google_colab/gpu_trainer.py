#!/usr/bin/env python3
"""
GPU-Accelerated Kill Prediction Trainer
Uses PyTorch for faster training on GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import argparse
from typing import Dict, List, Tuple
from enhanced_data_loader import EnhancedDataLoader
import platform
import time
import os
import torch_directml
import sqlite3
from database_data_loader import DatabaseDataLoader

# Force CPU usage
device = torch_directml.device()
print('Using device:', device)

def check_database_schema(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players';")
        result = cursor.fetchone()
        if not result:
            print("ERROR: 'players' table not found in the database.")
            conn.close()
            return False
        # Optionally, check for required columns
        cursor.execute("PRAGMA table_info(players);")
        columns = [row[1] for row in cursor.fetchall()]
        required_columns = [
            'name', 'team', 'rating', 'average_combat_score', 'kill_deaths',
            'kill_assists_survived_traded', 'average_damage_per_round',
            'kills_per_round', 'assists_per_round', 'first_kills_per_round',
            'first_deaths_per_round', 'headshot_percentage', 'clutch_success_percentage'
        ]
        missing = [col for col in required_columns if col not in columns]
        if missing:
            print(f"ERROR: Missing columns in 'players' table: {missing}")
            conn.close()
            return False
        conn.close()
        return True
    except Exception as e:
        print(f"ERROR: Could not check database schema: {e}")
        return False

class KillPredictionDataset(Dataset):
    """PyTorch dataset for kill prediction"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KillPredictionNN(nn.Module):
    """Neural network for kill prediction"""
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(KillPredictionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class GPUTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.data_loader = DatabaseDataLoader(db_path=os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db'))

    def prepare_data(self, limit_matches: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for GPU training using the database"""
        print("=== Preparing Data for GPU Training from Database ===")
        # Load player match data from the database
        df = self.data_loader.load_player_match_data()
        df = self.data_loader.calculate_player_features(df)
        # Prepare features and target
        X, y, feature_columns = self.data_loader.prepare_training_data(df)
        if X.size == 0 or y.size == 0:
            raise ValueError("No data available for training from database")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def train_neural_network(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                           X_test: torch.Tensor, y_test: torch.Tensor, X_df: pd.DataFrame) -> Dict:
        """Train neural network on GPU"""
        print("\n=== Training Neural Network on GPU ===")
        
        input_size = X_train.shape[1]
        model = KillPredictionNN(input_size, hidden_sizes=[256, 128, 64, 32]).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training parameters
        batch_size = 64
        epochs = 100
        early_stopping_patience = 15
        
        # Create data loaders
        train_dataset = KillPredictionDataset(X_train.cpu().numpy(), y_train.cpu().numpy())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test).squeeze()
                val_loss = criterion(val_outputs, y_test).item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_test_np = y_test.cpu().numpy()
        
        mse = mean_squared_error(y_test_np, y_pred)
        mae = mean_absolute_error(y_test_np, y_pred)
        r2 = r2_score(y_test_np, y_pred)
        
        print(f"\nFinal Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R2: {r2:.6f}")
        
        return {
            'model': model,
            'scaler': self.scaler,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_columns': list(X_df.columns),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def train_all_models(self, limit_matches: int = None):
        """Train all models using GPU acceleration"""
        print("=== GPU-Accelerated Kill Prediction Training ===")
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(limit_matches)
        
        # Get original X dataframe for feature columns
        # loader = EnhancedDataLoader() # This line is no longer needed as data is loaded directly
        # X, y = loader.prepare_training_data(limit_matches=limit_matches) # This line is no longer needed
        X = self.data_loader.load_player_match_data() # Load data directly from DB
        X = self.data_loader.calculate_player_features(X) # Calculate features
        X, y, feature_columns = self.data_loader.prepare_training_data(X) # Prepare features and target

        # Train neural network
        nn_result = self.train_neural_network(X_train, y_train, X_test, y_test, X)
        self.results['neural_network'] = nn_result
        
        # Save models
        self.save_models()
        
        # Generate report
        self.generate_training_report()
        
        return self.results
    
    def save_models(self):
        """Save trained models"""
        print("\n=== Saving Models ===")
        # Always save to the models directory inside this script's folder
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        for model_name, result in self.results.items():
            model_path = os.path.join(models_dir, f"{model_name}_gpu_model.pkl")
            # For PyTorch models, save state dict and other components
            if model_name == 'neural_network':
                model_data = {
                    'model_state_dict': result['model'].state_dict(),
                    'input_size': result['model'].network[0].in_features,
                    'hidden_sizes': [256, 128, 64, 32],  # Save the architecture
                    'scaler': result['scaler'],
                    'feature_columns': result['feature_columns'],
                    'performance': {
                        'mse': result['mse'],
                        'mae': result['mae'],
                        'r2': result['r2']
                    }
                }
            joblib.dump(model_data, model_path)
            print(f"Saved {model_name} model to {model_path}")
    
    def generate_training_report(self):
        """Generate training report"""
        print("\n=== Generating Training Report ===")
        # Always save to the models directory inside this script's folder
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        report = {
            'training_info': {
                'device_used': str(device),
                'models_trained': list(self.results.keys())
            },
            'model_performance': {}
        }
        for model_name, result in self.results.items():
            report['model_performance'][model_name] = {
                'mse': float(result['mse']),
                'mae': float(result['mae']),
                'r2': float(result['r2'])
            }
        # Save report
        report_path = os.path.join(models_dir, 'gpu_training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Training report saved to {report_path}")
        # Print summary
        print("\n=== Training Summary ===")
        for model_name, perf in report['model_performance'].items():
            print(f"{model_name}: MSE={perf['mse']:.6f}, MAE={perf['mae']:.6f}, R2={perf['r2']:.6f}")

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Kill Prediction Training')
    parser.add_argument('--limit-matches', type=int, default=None, 
                       help='Limit number of matches to use for training')
    args = parser.parse_args()

    # Database check before training
    db_path = os.path.join(os.path.dirname(__file__), '..', 'Scraper', 'valorant_matches.db')
    if not check_database_schema(db_path):
        print("Aborting training due to database issues.")
        return

    trainer = GPUTrainer()
    trainer.train_all_models(limit_matches=args.limit_matches)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)") 