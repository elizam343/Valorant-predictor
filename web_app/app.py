from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json
from datetime import datetime, timedelta
import pandas as pd

# Import our existing modules (these will need to be properly integrated)
# from kill_prediction_model.predictor import ValorantKillPredictor, KillLineBet
# from kill_prediction_model.data_loader import DataLoader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # TODO: Use environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///valorant_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    player_name = db.Column(db.String(100), nullable=False)
    team = db.Column(db.String(100), nullable=False)
    opponent_team = db.Column(db.String(100), nullable=False)
    kill_line = db.Column(db.Float, nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # OVER, UNDER, UNSURE
    confidence = db.Column(db.Float, nullable=False)
    map_name = db.Column(db.String(50))
    tournament = db.Column(db.String(100))
    match_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    actual_kills = db.Column(db.Integer)  # To be filled after match
    was_correct = db.Column(db.Boolean)  # To be filled after match

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    team1 = db.Column(db.String(100), nullable=False)
    team2 = db.Column(db.String(100), nullable=False)
    tournament = db.Column(db.String(100))
    match_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='scheduled')  # scheduled, live, completed
    map_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PlayerStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_name = db.Column(db.String(100), nullable=False)
    team = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Float)
    average_combat_score = db.Column(db.Float)
    kill_deaths = db.Column(db.Float)
    kills_per_round = db.Column(db.Float)
    assists_per_round = db.Column(db.Float)
    headshot_percentage = db.Column(db.Float)
    clutch_success_percentage = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Global variables (to be properly initialized)
predictor = None  # ValorantKillPredictor instance
data_loader = None  # DataLoader instance

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def initialize_services():
    """Initialize the prediction services"""
    global predictor, data_loader
    try:
        # TODO: Initialize predictor and data_loader
        # predictor = ValorantKillPredictor()
        # data_loader = DataLoader()
        pass
    except Exception as e:
        print(f"Error initializing services: {e}")

# Routes
@app.route('/')
def index():
    """Home page with overview"""
    return render_template('index.html')

@app.route('/predictions')
@login_required
def predictions():
    """View and create predictions"""
    # TODO: Implement prediction logic
    return render_template('predictions.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        # TODO: Implement prediction logic using the ML models
        # bet = KillLineBet(
        #     player_name=data['player_name'],
        #     team=data['team'],
        #     opponent_team=data['opponent_team'],
        #     kill_line=float(data['kill_line']),
        #     map=data.get('map', 'Unknown'),
        #     tournament=data.get('tournament', 'Unknown')
        # )
        # prediction = predictor.predict_kill_line(bet)
        
        # Placeholder response
        prediction = {
            'player_name': data['player_name'],
            'prediction': 'OVER',  # TODO: Use actual prediction
            'confidence': 0.75,    # TODO: Use actual confidence
            'recommendation': 'BET OVER'
        }
        
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/matches')
def matches():
    """View upcoming and past matches"""
    # TODO: Implement match listing
    return render_template('matches.html')

@app.route('/players')
def players():
    """View player statistics"""
    # TODO: Implement player listing with stats
    return render_template('players.html')

@app.route('/admin')
@login_required
def admin():
    """Admin panel for managing the system"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('index'))
    
    # TODO: Implement admin functionality
    return render_template('admin.html')

@app.route('/api/update-stats', methods=['POST'])
@login_required
def api_update_stats():
    """Update player statistics from scraper"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'})
    
    try:
        # TODO: Implement stats update from scraper
        # This should call the scraper to update player statistics
        return jsonify({'success': True, 'message': 'Stats updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train-model', methods=['POST'])
@login_required
def api_train_model():
    """Retrain the prediction model"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'})
    
    try:
        # TODO: Implement model training
        # This should retrain the ML models with new data
        return jsonify({'success': True, 'message': 'Model training completed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    # TODO: Implement login logic
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    # TODO: Implement registration logic
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        initialize_services()
    
    app.run(debug=True, host='0.0.0.0', port=5000) 