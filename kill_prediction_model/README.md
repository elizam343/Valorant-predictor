# Valorant Kill Line Prediction Model

A machine learning system for predicting whether Valorant players will go over or under their kill line in matches. The system uses player statistics from your existing database to make predictions with confidence levels.

## Features

- **Multi-Model Ensemble**: Combines Random Forest, Gradient Boosting, and Logistic Regression for robust predictions
- **Confidence-Based Predictions**: Marks predictions as "unsure" when confidence is too low
- **Comprehensive Analysis**: Player performance analysis, team comparisons, and historical tracking
- **Betting Reports**: Generate detailed reports for betting decisions
- **ROI Tracking**: Calculate return on investment for your predictions
- **Interactive Mode**: Command-line interface for making predictions

## Project Structure

```
kill_prediction_model/
├── __init__.py
├── data_loader.py      # Database connection and data preprocessing
├── models.py          # ML models and prediction logic
├── predictor.py       # Main prediction interface
├── utils.py           # Analysis and evaluation utilities
├── main.py            # Command-line interface and examples
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your Valorant player database is accessible from the Scraper folder.

## Quick Start

### Basic Usage

```python
from kill_prediction_model.predictor import ValorantKillPredictor, KillLineBet

# Initialize predictor
predictor = ValorantKillPredictor(use_ensemble=True)

# Create a betting opportunity
bet = KillLineBet("TenZ", "Sentinels", "Cloud9", 18.5, "Ascent", "VCT Champions")

# Make prediction
prediction = predictor.predict_kill_line(bet)
print(f"Recommendation: {prediction.recommended_action}")
print(f"Confidence: {prediction.confidence:.1%}")
```

### Training a Model

```python
# Prepare historical data
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
    }
    # Add more historical matches...
]

# Train the model
predictor = ValorantKillPredictor()
results = predictor.train_model(historical_data, save_path="models/kill_predictor.pkl")
```

### Command Line Usage

Run the demo to see all features:
```bash
python main.py --mode demo
```

Train a model:
```bash
python main.py --mode train --historical-data your_data.json
```

Interactive predictions:
```bash
python main.py --mode interactive --model-path models/kill_predictor.pkl
```

Generate a betting report:
```bash
python main.py --mode report --model-path models/kill_predictor.pkl
```

## Model Types

The system supports multiple machine learning approaches:

1. **Random Forest**: Good for handling non-linear relationships
2. **Gradient Boosting**: Excellent for complex patterns
3. **Logistic Regression**: Interpretable and fast
4. **Neural Network**: For deep learning approach
5. **Ensemble**: Combines all models for best performance

## Features Used

The model uses the following player statistics:
- Rating
- Average Combat Score (ACS)
- Kill/Death ratio
- Kills per round
- Assists per round
- First kills per round
- First deaths per round
- Headshot percentage
- Clutch success percentage
- Team average statistics
- Relative performance metrics

## Prediction Output

Each prediction includes:
- **Prediction Type**: OVER, UNDER, or UNSURE
- **Confidence Level**: 0-100% confidence in the prediction
- **Probabilities**: Individual probabilities for over/under/unsure
- **Recommended Action**: Clear betting recommendation

## Confidence Thresholds

- **High Confidence (≥70%)**: Strong betting recommendation
- **Medium Confidence (60-70%)**: Moderate recommendation
- **Low Confidence (<60%)**: Marked as "unsure" - avoid betting

## Historical Data Format

Your historical data should be in this format:
```json
[
  {
    "player_name": "Player Name",
    "team": "Team Name",
    "opponent_team": "Opponent Team",
    "kill_line": 18.5,
    "actual_kills": 22,
    "map": "Ascent",
    "tournament": "VCT Champions",
    "date": "2024-01-15"
  }
]
```

## Analysis Tools

### Player Analysis
```python
from kill_prediction_model.utils import DataAnalyzer
from kill_prediction_model.data_loader import DataLoader

data_loader = DataLoader()
analyzer = DataAnalyzer(data_loader)

# Get top players
top_players = analyzer.get_top_players('kills_per_round', 10)

# Analyze specific player
player_stats = predictor.analyze_player_history("TenZ", "Sentinels")
```

### ROI Calculation
```python
from kill_prediction_model.utils import ModelEvaluator

# Calculate ROI for your predictions
roi_results = ModelEvaluator.calculate_roi(predictions, actual_results, bet_amount=100)
print(f"ROI: {roi_results['roi']:.1%}")
```

## Model Evaluation

The system provides comprehensive evaluation metrics:
- Accuracy
- Classification reports
- Confusion matrices
- ROI calculations
- Confidence distributions

## Best Practices

1. **Data Quality**: Ensure your player database is up-to-date
2. **Historical Data**: Collect as much historical kill line data as possible
3. **Regular Retraining**: Retrain models with new data periodically
4. **Confidence Filtering**: Only bet on high-confidence predictions
5. **Bankroll Management**: Never bet more than you can afford to lose

## Troubleshooting

### Common Issues

1. **Player Not Found**: Make sure the player exists in your database
2. **Database Connection**: Verify the path to your SQLite database
3. **Missing Dependencies**: Install all required packages
4. **Low Confidence**: This is normal - the model is being conservative

### Getting Help

If you encounter issues:
1. Check that your database path is correct
2. Verify all dependencies are installed
3. Ensure your historical data format is correct
4. Check the console output for error messages

## Future Enhancements

Potential improvements for the system:
- Real-time data integration
- Advanced feature engineering
- Deep learning models
- Web interface
- API endpoints
- Mobile app integration

## License

This project is for educational and research purposes. Please use responsibly and in accordance with applicable laws and regulations regarding sports betting. 