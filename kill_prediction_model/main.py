#!/usr/bin/env python3
"""
Main script for Valorant Kill Line Prediction System

This script demonstrates how to use the prediction system to:
1. Train models with historical data
2. Make predictions on new kill lines
3. Generate betting reports
4. Analyze player performance
"""

import sys
import os
from typing import List, Dict
import argparse

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import ValorantKillPredictor, KillLineBet
from utils import ModelEvaluator, DataAnalyzer, HistoricalDataManager, export_sample_data
from data_loader import DataLoader

def train_model_example():
    """Example of training a model with sample data"""
    print("=== Training Model Example ===")
    
    # Create sample historical data
    historical_data = [
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
    
    # Initialize predictor
    predictor = ValorantKillPredictor(use_ensemble=True)
    
    # Train the model
    print("Training model with sample data...")
    results = predictor.train_model(historical_data, save_path="models/kill_predictor.pkl")
    
    print("Model training completed!")
    return predictor

def make_predictions_example(predictor: ValorantKillPredictor):
    """Example of making predictions on new kill lines"""
    print("\n=== Making Predictions Example ===")
    
    # Create betting opportunities
    bets = [
        KillLineBet("TenZ", "Sentinels", "Cloud9", 18.5, "Ascent", "VCT Champions"),
        KillLineBet("ShahZaM", "Sentinels", "Cloud9", 15.5, "Ascent", "VCT Champions"),
        KillLineBet("TenZ", "Sentinels", "Team Liquid", 16.5, "Haven", "VCT Champions"),
    ]
    
    # Make predictions
    predictions = predictor.predict_multiple_bets(bets)
    
    # Display results
    print("Prediction Results:")
    for pred in predictions:
        print(f"\n{pred.player_name} ({pred.team})")
        print(f"Kill Line: {pred.kill_line}")
        print(f"Prediction: {pred.recommended_action}")
        print(f"Confidence: {pred.confidence:.1%}")
        print(f"Over Probability: {pred.over_probability:.1%}")
        print(f"Under Probability: {pred.under_probability:.1%}")
    
    return predictions

def generate_report_example(predictor: ValorantKillPredictor):
    """Example of generating a comprehensive betting report"""
    print("\n=== Generating Betting Report Example ===")
    
    # Create multiple betting opportunities
    bets = [
        KillLineBet("TenZ", "Sentinels", "Cloud9", 18.5, "Ascent", "VCT Champions"),
        KillLineBet("ShahZaM", "Sentinels", "Cloud9", 15.5, "Ascent", "VCT Champions"),
        KillLineBet("TenZ", "Sentinels", "Team Liquid", 16.5, "Haven", "VCT Champions"),
        KillLineBet("ShahZaM", "Sentinels", "Team Liquid", 14.5, "Haven", "VCT Champions"),
    ]
    
    # Generate report
    report = predictor.generate_betting_report(bets, "example_betting_report.txt")
    print("Betting report generated and saved to example_betting_report.txt")
    
    return report

def analyze_players_example():
    """Example of analyzing player performance"""
    print("\n=== Player Analysis Example ===")
    
    data_loader = DataLoader()
    analyzer = DataAnalyzer(data_loader)
    
    # Get top players by kills per round
    print("Top 5 players by kills per round:")
    top_players = analyzer.get_top_players('kills_per_round', 5)
    print(top_players)
    
    # Analyze specific player
    print("\nAnalyzing TenZ's performance:")
    player_analysis = predictor.analyze_player_history("TenZ", "Sentinels")
    print(player_analysis)
    
    return top_players, player_analysis

def evaluate_model_example(predictions: List):
    """Example of evaluating model performance"""
    print("\n=== Model Evaluation Example ===")
    
    # Simulate actual results (in real usage, these would come from match outcomes)
    actual_results = [
        {'player_name': 'TenZ', 'actual_kills': 22, 'kill_line': 18.5},
        {'player_name': 'ShahZaM', 'actual_kills': 18, 'kill_line': 15.5},
        {'player_name': 'TenZ', 'actual_kills': 14, 'kill_line': 16.5},
    ]
    
    # Calculate ROI
    roi_results = ModelEvaluator.calculate_roi(predictions, actual_results, bet_amount=100)
    
    print("ROI Analysis:")
    print(f"Total Bets: ${roi_results['total_bets']}")
    print(f"Total Winnings: ${roi_results['total_winnings']:.2f}")
    print(f"Profit: ${roi_results['profit']:.2f}")
    print(f"ROI: {roi_results['roi']:.1%}")
    print(f"Accuracy: {roi_results['accuracy']:.1%}")
    
    return roi_results

def interactive_mode():
    """Interactive mode for making predictions"""
    print("\n=== Interactive Prediction Mode ===")
    
    predictor = ValorantKillPredictor()
    
    while True:
        print("\nEnter player and match details (or 'quit' to exit):")
        
        player_name = input("Player name: ").strip()
        if player_name.lower() == 'quit':
            break
        
        team = input("Team: ").strip()
        opponent = input("Opponent team: ").strip()
        
        try:
            kill_line = float(input("Kill line: ").strip())
        except ValueError:
            print("Invalid kill line. Please enter a number.")
            continue
        
        map_name = input("Map (optional): ").strip() or "Unknown"
        tournament = input("Tournament (optional): ").strip() or "Unknown"
        
        # Create bet and make prediction
        bet = KillLineBet(player_name, team, opponent, kill_line, map_name, tournament)
        
        try:
            prediction = predictor.predict_kill_line(bet)
            
            print(f"\nPrediction for {prediction.player_name}:")
            print(f"Recommendation: {prediction.recommended_action}")
            print(f"Confidence: {prediction.confidence:.1%}")
            print(f"Over Probability: {prediction.over_probability:.1%}")
            print(f"Under Probability: {prediction.under_probability:.1%}")
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("Make sure the player exists in the database.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Valorant Kill Line Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'report', 'analyze', 'interactive', 'demo'],
                       default='demo', help='Mode to run')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--historical-data', type=str, help='Path to historical data file')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training mode - requires historical data")
        if args.historical_data:
            # Load historical data from file
            import json
            with open(args.historical_data, 'r') as f:
                historical_data = json.load(f)
            
            predictor = ValorantKillPredictor()
            predictor.train_model(historical_data, save_path="models/kill_predictor.pkl")
        else:
            print("Please provide historical data file with --historical-data")
    
    elif args.mode == 'predict':
        print("Prediction mode")
        predictor = ValorantKillPredictor(model_path=args.model_path)
        interactive_mode()
    
    elif args.mode == 'report':
        print("Report generation mode")
        predictor = ValorantKillPredictor(model_path=args.model_path)
        generate_report_example(predictor)
    
    elif args.mode == 'analyze':
        print("Analysis mode")
        analyze_players_example()
    
    elif args.mode == 'interactive':
        print("Interactive mode")
        interactive_mode()
    
    elif args.mode == 'demo':
        print("Running full demo...")
        
        # Export sample data
        export_sample_data()
        
        # Run all examples
        try:
            predictor = train_model_example()
            predictions = make_predictions_example(predictor)
            generate_report_example(predictor)
            analyze_players_example()
            evaluate_model_example(predictions)
            
            print("\n=== Demo Completed Successfully! ===")
            print("Check the generated files:")
            print("- example_betting_report.txt")
            print("- sample_historical_data.json")
            print("- models/kill_predictor.pkl")
            
        except Exception as e:
            print(f"Demo failed with error: {e}")
            print("This might be due to missing database or dependencies.")
            print("Make sure you have the Scraper database set up.")

if __name__ == "__main__":
    main() 