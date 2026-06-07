# 🚀 Valorant Kill Predictor - Setup Guide

This guide will help you set up the complete web application with your trained prediction model.

## 📋 Prerequisites

- Python 3.8+
- Your trained model files (from Google Colab)
- Valorant match database
- Flask and dependencies

## 🛠️ Setup Steps

### 1. Copy Your Trained Model

Copy your best model from Google Colab to the web_app directory:

```bash
# Copy your precision model (recommended)
cp /path/to/precision_model.pkl web_app/models/

# Or copy stable model
cp /path/to/stable_model.pkl web_app/models/

# Ensure the database is accessible
cp /path/to/clean_valorant_matches.db Scraper/valorant_matches.db
```

### 2. Install Dependencies

```bash
cd web_app
pip install -r requirements.txt

# Additional dependencies for PyTorch
pip install torch torchvision torchaudio
```

### 3. Directory Structure

Ensure your directory structure looks like this:

```
Valorant-predictor/
├── web_app/
│   ├── app.py
│   ├── prediction_service.py
│   ├── models/
│   │   └── precision_model.pkl  # Your trained model
│   ├── templates/
│   └── static/
├── Scraper/
│   └── valorant_matches.db      # Your database
└── google_colab/
    └── (your training notebooks)
```

### 4. Run the Application

```bash
cd web_app
python app.py
```

The application will:
- ✅ Auto-discover your trained model
- ✅ Connect to your database
- ✅ Start the web server on http://localhost:5000

## 🎯 Features Available

### For Users
- **Kill Prediction Interface**: Enter player name, get AI predictions
- **Autocomplete**: Search for players from your database
- **Confidence Analysis**: Every prediction includes confidence levels
- **Betting Recommendations**: Get over/under advice with kill lines
- **Match Context**: Specify opponent, map, series type, tournament

### For Predictions
- **Player Stats**: Recent averages, historical performance
- **Momentum Analysis**: Form trends and consistency
- **Role Classification**: Automatic player role detection
- **Experience Weighting**: More confident predictions for experienced players

## 🔧 Configuration

### Model Priority
The system automatically loads models in this order:
1. `models/precision_model.pkl` (best performance)
2. `models/stable_model.pkl` (reliable)
3. `models/gpu_maximized_model.pkl` (experimental)

### Database Path
Update the database path in `app.py` if needed:
```python
predictor = ValorantKillPredictor(
    model_path=None,  # Auto-discover
    db_path="../Scraper/your_database_name.db"  # Update this
)
```

## 📊 Usage Examples

### Basic Prediction
1. Go to http://localhost:5000/predict
2. Enter player name (e.g., "TenZ")
3. Optional: Add opponent team, map, series type
4. Click "Predict Kills"

### Betting Analysis
1. Enter all match details
2. Add the bookmaker's kill line (e.g., 15.5)
3. Get over/under recommendation with reasoning

### API Usage
```javascript
// Make prediction via API
fetch('/api/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        player_name: 'TenZ',
        opponent_team: 'Team Liquid',
        map_name: 'Ascent',
        series_type: 'bo3',
        tournament: 'masters',
        kill_line: 18.5
    })
})
```

## 🎯 Production Deployment

### Security Updates
Before deploying to production:

1. **Change secret key**:
```python
app.config['SECRET_KEY'] = 'your-secure-random-key'
```

2. **Use environment variables**:
```python
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
```

3. **Configure database properly**:
```python
# Use PostgreSQL or MySQL for production
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
```

### Deployment Options
- **Local**: Run with `python app.py`
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **Docker**: Create `Dockerfile` with Python and dependencies
- **VPS**: Use nginx + gunicorn for production

## 🚨 Troubleshooting

### Model Not Loading
```
⚠️ No trained model found
```
**Solution**: Ensure model file is in `web_app/models/` directory

### Database Not Found
```
⚠️ Database not found at: ../Scraper/valorant_matches.db
```
**Solution**: Check database path and ensure file exists

### Player Not Found
```
Player "xyz" not found in database
```
**Solution**: Check spelling or try partial name matching

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Install PyTorch: `pip install torch`

## 📈 Performance Tips

### For Better Predictions
1. **Keep database updated** with recent matches
2. **Use precise player names** from your database
3. **Provide context** (opponent, map, tournament) for better accuracy
4. **Check confidence levels** - only bet on high-confidence predictions

### For Better Performance
1. **Model caching**: Model loads once and stays in memory
2. **Database indexing**: Ensure player names are indexed
3. **Batch predictions**: Use API for multiple predictions

## 🎉 Success Indicators

When everything is working correctly, you should see:

```
🔄 Loading model from: models/precision_model.pkl
✅ Loaded Precision-Tuned model
📊 Model performance: 3.287 MAE
✅ Services initialized successfully
🚀 Starting Valorant Kill Predictor web application...
```

Your application is now ready for production use with:
- ✅ Professional-grade 3.3 MAE accuracy
- ✅ Stable, bounded predictions (0-35 kills)
- ✅ User-friendly web interface
- ✅ API endpoints for integration
- ✅ Confidence-based recommendations

## 🎯 Next Steps

1. **Test with known players** from your database
2. **Compare predictions** with actual match results
3. **Track betting performance** if using for wagering
4. **Consider implementing** user accounts for prediction history
5. **Scale up** with more sophisticated deployment if needed

Your Valorant kill prediction system is now production-ready! 🏆 