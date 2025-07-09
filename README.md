# Valorant Kill Line Predictor

A comprehensive machine learning system for predicting Valorant player kill lines in esports matches, featuring data collection, ML models, and web interface components.

## Project Overview

This project consists of three main components that need to be integrated into a complete system:

1. **Data Scraper** (`Scraper/`) - Collects player statistics from VLR.gg
2. **ML Prediction System** (`kill_prediction_model/`) - Machine learning models for predictions
3. **Web Application** (`web_app/`) - Flask-based web interface (skeleton)

## Current State

### ✅ What's Working
- **Scraper**: 90% complete - collects player data from VLR.gg and stores in SQLite
- **ML Framework**: Complete structure with ensemble models (Random Forest, Gradient Boosting, Logistic Regression)
- **Web App Skeleton**: Modern Flask application with authentication framework and UI

### ❌ What's Missing (Integration Challenges)
- **System Integration**: The three components are completely disconnected
- **ML Model Training**: Models exist but aren't trained on real data
- **Web App Functionality**: Authentication, API endpoints, and real-time features not implemented
- **Database Integration**: Web app doesn't connect to scraper database
- **Production Features**: Error handling, testing, security, deployment

## Project Structure

```
valorant-kill-line-predictor/
├── Scraper/                    # Data collection system
│   ├── app.py                 # Flask web interface for data browsing
│   ├── scraper_api.py         # VLR.gg API integration
│   ├── db_utils.py            # Database utilities
│   ├── vlr_players.db         # SQLite database with player stats
│   └── templates/             # Web templates for data viewing
├── kill_prediction_model/      # Machine learning system
│   ├── predictor.py           # Main prediction interface
│   ├── models.py              # ML models (Random Forest, Gradient Boosting, etc.)
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── utils.py               # Analysis and evaluation utilities
│   └── main.py                # Command-line interface
├── web_app/                   # Web application (skeleton)
│   ├── app.py                 # Flask application with routes
│   ├── templates/             # HTML templates
│   └── requirements.txt       # Python dependencies
├── switch.py                  # Git branch management for DataAnnotation
└── README.md                  # This file
```

## Key Integration Challenges

### 1. Database Integration
- Connect web app to existing `Scraper/vlr_players.db`
- Sync data between scraper and web app databases
- Handle database migrations and schema updates

### 2. ML Model Integration
- Train models using actual player data from scraper
- Integrate `ValorantKillPredictor` class into web app
- Create API endpoints for making predictions
- Implement model retraining and updates

### 3. Web Application Completion
- Implement user authentication and registration
- Create functional API endpoints
- Add real-time dashboard updates
- Implement admin panel functionality
- Add proper error handling and validation

### 4. System Architecture
- Design proper data flow between components
- Implement secure authentication
- Add logging and monitoring
- Create comprehensive testing suite

## Technical Requirements

### Dependencies
- **Python 3.8+**
- **Flask** for web application
- **SQLAlchemy** for database management
- **scikit-learn** for machine learning
- **pandas/numpy** for data processing
- **requests** for API calls

### Database Schema
The system uses multiple databases:
- `Scraper/vlr_players.db` - Player statistics from VLR.gg
- `web_app/valorant_predictions.db` - User data, predictions, matches

### ML Models
- **Random Forest** - Good for non-linear relationships
- **Gradient Boosting** - Excellent for complex patterns
- **Logistic Regression** - Interpretable and fast
- **Ensemble** - Combines all models for best performance

## Development Tasks

### High Priority
1. **Fix imports and module integration**
2. **Connect web app to scraper database**
3. **Train ML models with real data**
4. **Implement authentication system**
5. **Create functional prediction API**

### Medium Priority
1. **Add admin panel functionality**
2. **Implement real-time updates**
3. **Add comprehensive testing**
4. **Improve error handling**
5. **Add data validation**

### Low Priority
1. **Performance optimization**
2. **Advanced UI features**
3. **Analytics and reporting**
4. **Mobile responsiveness**
5. **Deployment automation**

## Usage Examples

### Training Models
```python
from kill_prediction_model.predictor import ValorantKillPredictor
from kill_prediction_model.data_loader import DataLoader

# Load data from scraper database
data_loader = DataLoader()
historical_data = data_loader.load_training_data()

# Train ensemble model
predictor = ValorantKillPredictor(use_ensemble=True)
results = predictor.train_model(historical_data)
```

### Making Predictions
```python
from kill_prediction_model.predictor import KillLineBet

# Create prediction request
bet = KillLineBet("TenZ", "Sentinels", "Cloud9", 18.5, "Ascent", "VCT Champions")
prediction = predictor.predict_kill_line(bet)

print(f"Prediction: {prediction.recommended_action}")
print(f"Confidence: {prediction.confidence:.1%}")
```

### Web Application
```bash
cd web_app
pip install -r requirements.txt
python app.py
```

## Security Considerations

- **Authentication**: Implement secure user registration and login
- **Input Validation**: Validate all user inputs and API requests
- **CSRF Protection**: Add CSRF tokens to forms
- **Rate Limiting**: Implement API rate limiting
- **Error Handling**: Avoid exposing sensitive information in errors

## Testing Strategy

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test system integration points
- **End-to-End Tests**: Test complete user workflows
- **Security Tests**: Test authentication and authorization
- **Performance Tests**: Test database and ML model performance

## Deployment

### Development
```bash
python web_app/app.py
```

### Production Considerations
- Use environment variables for configuration
- Set up proper database backups
- Configure web server (nginx, Apache)
- Enable SSL/TLS encryption
- Set up monitoring and logging
- Implement automated testing

## Contributing

This project is designed for educational and research purposes. Please ensure all code follows best practices and includes proper documentation.

## License

This project is for educational purposes. Please use responsibly and in accordance with applicable laws and regulations regarding sports betting.

## Support

For issues or questions, please check the individual component README files or create an issue in the repository.

---

**Note**: This project is currently in development and requires significant integration work to become fully functional. The complexity of connecting multiple systems makes it an excellent challenge for AI-assisted development. 