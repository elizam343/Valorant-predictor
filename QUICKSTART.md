# Quick Start Guide

Get the Valorant Kill Line Predictor up and running in 5 minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (One-time setup)
```bash
cd kill_prediction_model
python gpu_trainer.py --limit-matches 5000
```

### 3. Test a Prediction
```bash
python advanced_matchup_predictor.py
```

### 4. Run the Web App
```bash
cd ../web_app
python app.py
```

## 🎯 Example Usage

### Python API
```python
from kill_prediction_model.advanced_matchup_predictor import AdvancedMatchupPredictor, MatchupContext

# Initialize predictor
predictor = AdvancedMatchupPredictor()

# Create matchup
matchup = MatchupContext(
    player_name="aspas",
    player_team="MIBR",
    opponent_team="FUR", 
    tournament="VCT Champions",
    series_type="bo3",
    maps=["Ascent", "Haven"],
    kill_line=0.85
)

# Get prediction
result = predictor.predict_matchup(matchup)
print(f"Prediction: {result.predicted_kills_per_round:.3f} kills/round")
print(f"Recommendation: {result.recommendation}")
```

### Web Interface
1. Open `http://localhost:5000` in your browser
2. Select player, team, opponent, and tournament
3. Enter kill line and maps
4. Get instant prediction with confidence intervals

## 📊 What You Get

- **Predicted kills per round** for specific matchups
- **Confidence intervals** (80%, 90%, 95%)
- **Statistical significance** testing
- **Smart recommendations** (OVER/UNDER/UNSURE)
- **Detailed reasoning** for each prediction

## 🔧 Troubleshooting

### Model Training Issues
- Ensure you have enough disk space for 5,000+ matches
- Check that PyTorch is installed correctly
- Verify internet connection for data scraping

### Web App Issues  
- Make sure Flask is installed: `pip install Flask`
- Check port 5000 is available
- Verify all dependencies are installed

### Prediction Issues
- Ensure model is trained first
- Check player name exists in database
- Verify all matchup parameters are provided

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Open an issue on GitHub for bugs
- Review the API documentation in the code

---

**Ready to predict! 🎮** 