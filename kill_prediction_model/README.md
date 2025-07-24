# Valorant Kill Line Prediction Model

A machine learning system for predicting whether Valorant players will go over or under their kill line in matches. The system now uses series-level (per-match) player statistics, aggregating performance across maps, to make more context-aware predictions.

## Features

- **Series-Level Aggregation**: Learns from each match (series), capturing player performance across maps and map order effects
- **Context-Rich Features**: Includes map order, agent selection, and team context for each map in a series
- **Neural Network Training**: Supports deep learning models for richer pattern recognition
- **Multi-Model Ensemble**: (Legacy) Combines Random Forest, Gradient Boosting, and Logistic Regression for robust predictions
- **Confidence-Based Predictions**: Marks predictions as "unsure" when confidence is too low
- **Comprehensive Analysis**: Player performance analysis, team comparisons, and historical tracking
- **Betting Reports**: Generate detailed reports for betting decisions
- **ROI Tracking**: Calculate return on investment for your predictions

## Project Structure

```
kill_prediction_model/
├── __init__.py
├── data_loader.py      # (Legacy) Database connection and data preprocessing
├── enhanced_data_loader.py # Series-level data aggregation and feature engineering
├── models.py          # ML models and prediction logic
├── gpu_trainer.py     # Neural network training with series-level data
├── advanced_matchup_predictor.py # Series-level prediction logic
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

## Series-Level Data Pipeline (New)

The new pipeline aggregates player stats per match (series), capturing:
- Player performance for each map in order (first, second, etc.)
- Agent played, map name, and side for each map
- Series type (bo1, bo3, etc.), tournament, teams, and date
- Target variable: total kills in the first two maps (for kill line bets)

### Example Feature Vector
- kills_map1, kills_map2, agent_map1, agent_map2, map1_name, map2_name, side_map1, side_map2, ...
- series_type, tournament, player_name, team, opponent_team
- total_kills_first2 (target)

### Data Preparation
The `EnhancedDataLoader` class provides `create_series_level_dataset()` to produce a DataFrame with one row per player per match, with all relevant features and the target.

## Training a Model (New)

Train a neural network on the new series-level dataset:
```bash
python kill_prediction_model/gpu_trainer.py --limit-matches 1000
```
- The model and training report are saved in `kill_prediction_model/models/`.
- You can increase `--limit-matches` or remove it to use all available data.

## Prediction (New)

Use `advanced_matchup_predictor.py` to make series-level predictions:
```bash
python kill_prediction_model/advanced_matchup_predictor.py
```
- Edit the script to specify the player, series type, and kill line for the first two maps.

## Legacy Usage

The old per-map logic and ensemble models are still available for reference, but the recommended approach is to use the new series-level pipeline for both training and prediction.

## Model Types

- **Neural Network**: Learns from series-level, context-rich features (recommended)
- **Random Forest, Gradient Boosting, Logistic Regression**: (Legacy) For comparison and ensemble

## Features Used (New)

- Kills, deaths, assists per map (ordered)
- Agent played per map
- Map name and order
- Side (attack/defense) per map
- Series type, tournament, teams
- Aggregated/cumulative stats for first two maps

## Prediction Output

Each prediction includes:
- **Prediction Type**: OVER, UNDER, or UNSURE
- **Confidence Level**: 0-100% confidence in the prediction
- **Probabilities**: Individual probabilities for over/under/unsure
- **Recommended Action**: Clear betting recommendation

## Historical Data Format

No change, but now the model expects series-level aggregation. See the new data loader for details.

## Troubleshooting

- If you see errors about missing directories, ensure `kill_prediction_model/models/` exists.
- If you see NaN predictions, check that the player and match context exist in your data and that all features are present.

## Future Enhancements

- Add team map preferences, agent pool, and side performance as features
- Support for new maps and evolving meta
- Real-time data integration and web/API interface

## License

This project is for educational and research purposes. Please use responsibly and in accordance with applicable laws and regulations regarding sports betting. 