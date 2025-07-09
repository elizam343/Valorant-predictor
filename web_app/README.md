# Valorant Kill Line Predictor - Web Application

This is a **skeleton web application** for the Valorant Kill Line Predictor system. The application is designed to provide a modern web interface for making kill line predictions, but is currently incomplete and requires significant development work.

## Current State

### ✅ What's Implemented (Skeleton)
- Flask application structure with proper routing
- Database models for users, predictions, matches, and player stats
- Authentication system framework (Flask-Login)
- Modern responsive UI with Bootstrap 5
- API endpoints structure for predictions and admin functions
- Template structure with base layout

### ❌ What's Missing (Needs Implementation)
- **Integration with existing ML models** - The web app needs to connect to the `kill_prediction_model` package
- **Database integration** - Need to connect with the existing scraper database
- **Authentication logic** - Login/register functionality is not implemented
- **API functionality** - All API endpoints return placeholder data
- **Real-time data** - Dashboard shows static placeholder data
- **Admin panel** - No actual admin functionality implemented
- **Error handling** - Basic error pages exist but no real error handling
- **Testing** - No tests implemented

## Architecture

```
web_app/
├── app.py                 # Main Flask application (skeleton)
├── templates/             # HTML templates (complete structure)
│   ├── base.html         # Base template with navigation
│   └── index.html        # Dashboard page
├── static/               # Static files (CSS, JS, images)
├── requirements.txt      # Python dependencies
└── README.md            # This file

Integration Points:
├── kill_prediction_model/  # ML prediction system
├── Scraper/               # Data collection system
└── Database integration   # SQLite databases
```

## Key Integration Challenges

### 1. ML Model Integration
The web app needs to properly import and use the existing `ValorantKillPredictor` class:
```python
# TODO: Fix imports and integration
from kill_prediction_model.predictor import ValorantKillPredictor, KillLineBet
from kill_prediction_model.data_loader import DataLoader
```

### 2. Database Integration
Need to connect the web app's SQLAlchemy models with the existing scraper database:
- Import player data from `Scraper/vlr_players.db`
- Sync data between systems
- Handle database migrations

### 3. Authentication System
Implement proper user authentication:
- User registration and login
- Password hashing and security
- Session management
- Role-based access control

### 4. API Development
Complete all API endpoints:
- `/api/predict` - Make predictions using ML models
- `/api/update-stats` - Update player statistics
- `/api/train-model` - Retrain ML models
- Dashboard data endpoints

### 5. Real-time Features
Implement dynamic functionality:
- Live dashboard updates
- Real-time prediction results
- System status monitoring
- Data refresh capabilities

## Development Tasks

### High Priority
1. **Fix imports and model integration**
2. **Implement authentication system**
3. **Connect to existing databases**
4. **Complete API endpoints**
5. **Add proper error handling**

### Medium Priority
1. **Implement admin panel functionality**
2. **Add user management features**
3. **Create comprehensive testing suite**
4. **Add logging and monitoring**
5. **Implement data validation**

### Low Priority
1. **Add advanced UI features**
2. **Implement real-time updates**
3. **Add export functionality**
4. **Create mobile-responsive design**
5. **Add analytics and reporting**

## Running the Application

### Prerequisites
- Python 3.8+
- All dependencies from `requirements.txt`
- Access to existing `kill_prediction_model` and `Scraper` directories

### Installation
```bash
cd web_app
pip install -r requirements.txt
```

### Running (Development)
```bash
python app.py
```

**Note**: The application will start but most functionality will not work due to incomplete implementation.

## Security Considerations

The current skeleton includes several security issues that need to be addressed:

1. **Hardcoded secret key** - Should use environment variables
2. **No input validation** - All forms need proper validation
3. **No CSRF protection** - Forms need CSRF tokens
4. **No rate limiting** - API endpoints need rate limiting
5. **No proper error handling** - Errors could expose sensitive information

## Testing Strategy

The application needs comprehensive testing:

1. **Unit tests** for all models and utilities
2. **Integration tests** for API endpoints
3. **End-to-end tests** for user workflows
4. **Security tests** for authentication and authorization
5. **Performance tests** for database and ML model operations

## Deployment Considerations

For production deployment, consider:

1. **Environment configuration** - Use environment variables
2. **Database setup** - Proper database configuration
3. **Static file serving** - Configure web server for static files
4. **SSL/TLS** - Secure communication
5. **Monitoring** - Application and system monitoring
6. **Backup strategy** - Database and model backups

## Next Steps

This skeleton provides a solid foundation for a modern web application. The next phase should focus on:

1. **Integration** - Connect all existing components
2. **Functionality** - Implement core features
3. **Testing** - Ensure reliability and security
4. **Deployment** - Prepare for production use

The complexity of integrating multiple systems (scraper, ML models, web interface) makes this an excellent challenge for AI models to complete. 