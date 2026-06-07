import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import pandas as pd
import json

# Add the project directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'kill_prediction_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scraper'))

# Import our ML prediction system
from kill_prediction_model.predictor import ValorantKillPredictor, KillLineBet
from kill_prediction_model.data_loader import DataLoader
from Scraper.db_utils import get_players, get_teams, get_players_by_team
from Scraper.team_mapping import get_full_team_name, get_display_name

app = Flask(__name__, template_folder='web_app/templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
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
    opponent_team = db.Column(db.String(100))
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

# Global variables for ML system
predictor = None
data_loader = None

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def initialize_services():
    """Initialize the prediction services"""
    global predictor, data_loader
    try:
        print("Initializing ML prediction services...")
        data_loader = DataLoader()
        predictor = ValorantKillPredictor(use_ensemble=True)
        
        # Auto-train the model with sample data on startup
        print("Training model with sample data...")
        try:
            historical_data = create_sample_historical_data()
            results = predictor.train_model(historical_data)
            print(f"Model trained successfully with accuracy: {results.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
        
        print("ML services initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing services: {e}")
        return False

def generate_prediction_explanation(prediction_result, player_analysis, bet):
    """Generate detailed explanation for the prediction"""
    explanation = []
    
    # Player performance analysis
    stats = player_analysis.get('stats', {})
    percentiles = player_analysis.get('percentiles', {})
    strengths = player_analysis.get('strengths', [])
    weaknesses = player_analysis.get('weaknesses', [])
    
    # Basic player info
    explanation.append(f"**Player Analysis for {prediction_result.player_name}:**")
    
    # Rating analysis
    rating = stats.get('rating', 0)
    rating_percentile = percentiles.get('rating', 50)
    if rating_percentile >= 75:
        explanation.append(f"• High-rated player ({rating:.2f} rating, {rating_percentile:.0f}th percentile) - typically performs well")
    elif rating_percentile <= 25:
        explanation.append(f"• Lower-rated player ({rating:.2f} rating, {rating_percentile:.0f}th percentile) - may struggle in tough matchups")
    else:
        explanation.append(f"• Average-rated player ({rating:.2f} rating, {rating_percentile:.0f}th percentile)")
    
    # Kills per round analysis
    kpr = stats.get('kills_per_round', 0)
    kpr_percentile = percentiles.get('kills_per_round', 50)
    # Determine number of maps for estimate
    if bet.series_type == "BO1":
        maps_count = 1
    elif bet.series_type == "BO3":
        maps_count = 2 if bet.maps_scope == "first_2" else 3
    elif bet.series_type == "BO5":
        maps_count = 3 if bet.maps_scope == "first_3" else 5
    else:
        maps_count = 2
    estimated_kills = kpr * 24 * maps_count  # 24 rounds per map
    explanation.append(f"• Averages {kpr:.2f} kills per round (~{estimated_kills:.0f} kills in {maps_count} map{'s' if maps_count > 1 else ''})")
    
    # Kill line comparison
    kill_line = bet.kill_line
    gap = estimated_kills - kill_line
    if gap > 3:
        explanation.append(f"• Kill line ({kill_line}) is much lower than expected output for {maps_count} map{'s' if maps_count > 1 else ''}")
    elif gap < -3:
        explanation.append(f"• Kill line ({kill_line}) is much higher than expected output for {maps_count} map{'s' if maps_count > 1 else ''}")
    else:
        explanation.append(f"• Kill line ({kill_line}) is close to expected output for {maps_count} map{'s' if maps_count > 1 else ''}")
    
    # Series format impact
    if bet.series_type == "BO1":
        explanation.append(f"• BO1 format: Single map performance - higher variance expected")
    elif bet.series_type == "BO3":
        if bet.maps_scope == "first_2":
            explanation.append(f"• BO3, first 2 maps: Focused on early series performance")
        else:
            explanation.append(f"• BO3 format: Multiple maps allow for consistency")
    elif bet.series_type == "BO5":
        if bet.maps_scope == "first_3":
            explanation.append(f"• BO5, first 3 maps: Focused on early series performance")
        else:
            explanation.append(f"• BO5 format: Extended series favors consistent performers")
    
    # Strengths and weaknesses
    if strengths:
        explanation.append(f"• **Strengths**: {', '.join(strengths)}")
    if weaknesses:
        explanation.append(f"• **Weaknesses**: {', '.join(weaknesses)}")
    
    # Prediction reasoning
    explanation.append(f"\n**Prediction Reasoning:**")
    
    if prediction_result.prediction.name == "OVER":
        explanation.append(f"• Model predicts OVER {kill_line} kills")
        explanation.append(f"• {prediction_result.over_probability:.1%} probability of going over")
        if kpr_percentile >= 75:
            explanation.append(f"• High kill rate supports over prediction")
        if rating_percentile >= 75:
            explanation.append(f"• Strong overall rating supports consistent performance")
    elif prediction_result.prediction.name == "UNDER":
        explanation.append(f"• Model predicts UNDER {kill_line} kills")
        explanation.append(f"• {prediction_result.under_probability:.1%} probability of going under")
        if kpr_percentile <= 25:
            explanation.append(f"• Lower kill rate supports under prediction")
        # If the average is above the kill line but the model predicts UNDER, clarify why
        if estimated_kills > kill_line and prediction_result.under_probability > 0.7:
            explanation.append(f"• Although the average output suggests {estimated_kills:.0f} kills in {maps_count} map{'s' if maps_count > 1 else ''}, the model predicts UNDER due to other risk factors (recent form, opponent strength, or model confidence).")
        # Only use the "rarely exceeds" line if the average is below the kill line or if confidence is very high
        if gap < -3 and prediction_result.confidence >= 0.9:
            explanation.append(f"• The model is highly confident because {prediction_result.player_name} rarely exceeds {kill_line} kills in {maps_count} map{'s' if maps_count > 1 else ''}.")
        elif prediction_result.confidence >= 0.7 and gap < -3:
            explanation.append(f"• The model is confident due to the large gap between average and kill line.")
    else:
        explanation.append(f"• Model is UNSURE - avoid betting")
        explanation.append(f"• Confidence too low ({prediction_result.confidence:.1%}) for reliable prediction")
    
    # Confidence explanation
    if prediction_result.confidence >= 0.8:
        explanation.append(f"• **High confidence** ({prediction_result.confidence:.1%}) - Strong betting opportunity")
    elif prediction_result.confidence >= 0.6:
        explanation.append(f"• **Moderate confidence** ({prediction_result.confidence:.1%}) - Reasonable betting opportunity")
    else:
        explanation.append(f"• **Low confidence** ({prediction_result.confidence:.1%}) - Avoid betting")
    
    # Map and opponent factors
    if bet.map and bet.map != "Unknown" and bet.map.strip():
        explanation.append(f"• Map: {bet.map} - considers player's map-specific performance")
    if bet.opponent_team and bet.opponent_team != "Unknown" and bet.opponent_team.strip():
        explanation.append(f"• vs {bet.opponent_team} - considers head-to-head and matchup context")
    
    return "\n".join(explanation)

def create_sample_historical_data():
    """Create realistic historical data based on actual player statistics"""
    try:
        players = get_players()[:400]  # Get first 400 players for much more data
        sample_data = []
        
        for i, player in enumerate(players):
            # Extract player stats for realistic kill predictions
            try:
                # Parse player stats (handle string values)
                kills_per_round = float(player[8]) if player[8] and player[8] != 'N/A' else 0.8
                rating = float(player[3]) if player[3] and player[3] != 'N/A' else 1.0
                
                # Estimate realistic kill range based on player stats
                # Higher rated players typically get more kills
                base_kills_estimate = max(10, min(30, kills_per_round * 20 + rating * 5))
                
                # Create varied training examples based on different scenarios
                scenarios = [
                    # Good performance scenarios (over) - 40% of examples
                    {'kill_line': base_kills_estimate - 2.5, 'actual_kills': int(base_kills_estimate + 3), 'performance': 'good'},
                    {'kill_line': base_kills_estimate - 1.5, 'actual_kills': int(base_kills_estimate + 2), 'performance': 'good'},
                    {'kill_line': base_kills_estimate - 0.5, 'actual_kills': int(base_kills_estimate + 1), 'performance': 'good'},
                    {'kill_line': base_kills_estimate + 0.5, 'actual_kills': int(base_kills_estimate + 2), 'performance': 'good'},
                    
                    # Poor performance scenarios (under) - 40% of examples
                    {'kill_line': base_kills_estimate + 1.5, 'actual_kills': int(base_kills_estimate - 2), 'performance': 'poor'},
                    {'kill_line': base_kills_estimate + 2.5, 'actual_kills': int(base_kills_estimate - 3), 'performance': 'poor'},
                    {'kill_line': base_kills_estimate + 3.5, 'actual_kills': int(base_kills_estimate - 1), 'performance': 'poor'},
                    {'kill_line': base_kills_estimate + 1.0, 'actual_kills': int(base_kills_estimate - 1), 'performance': 'poor'},
                    
                    # Average performance scenarios (mixed) - 20% of examples
                    {'kill_line': base_kills_estimate + 0.5, 'actual_kills': int(base_kills_estimate), 'performance': 'average'},
                    {'kill_line': base_kills_estimate - 0.5, 'actual_kills': int(base_kills_estimate), 'performance': 'average'},
                ]
                
                # Generate training examples for different maps and opponents
                maps = ['Ascent', 'Bind', 'Haven', 'Split', 'Icebox', 'Breeze']
                tournaments = ['VCT Champions', 'VCT Masters', 'VCT Challengers', 'Regional League']
                
                for j, scenario in enumerate(scenarios):
                    if scenario['kill_line'] > 0 and scenario['actual_kills'] > 0:  # Ensure positive values
                        sample_data.append({
                            'player_name': player[1],
                            'team': player[2],
                            'opponent_team': f'Team_{(i+j)%10}',  # Vary opponents
                            'kill_line': scenario['kill_line'],
                            'actual_kills': scenario['actual_kills'],
                            'map': maps[j % len(maps)],
                            'tournament': tournaments[j % len(tournaments)],
                            'date': f'2024-01-{15 + (j % 15):02d}',
                            'performance_type': scenario['performance']
                        })
                        
            except (ValueError, TypeError, IndexError):
                # Skip players with invalid data
                continue
        
        print(f"Generated {len(sample_data)} realistic training examples from {len(players)} players")
        return sample_data
        
    except Exception as e:
        print(f"Error creating training data: {e}")
        # Fallback to basic sample data
        return [
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
            }
        ] * 50  # Repeat for more examples

# Routes
@app.route('/')
def index():
    """Home page with overview"""
    try:
        # Get basic stats
        total_players = len(get_players())
        total_teams = len(get_teams())
        total_predictions = Prediction.query.count()
        
        # Get recent predictions
        recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
        
        stats = {
            'total_players': total_players,
            'total_teams': total_teams,
            'total_predictions': total_predictions,
            'recent_predictions': recent_predictions
        }
        
        return render_template('index.html', stats=stats)
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        return render_template('index.html', stats={})

@app.route('/predictions')
@login_required
def predictions():
    """View and create predictions"""
    try:
        # Get all teams for the dropdown and convert to full names
        teams_raw = get_teams()
        teams_with_full_names = []
        
        for team in teams_raw:
            full_name = get_display_name(team)
            # Only include teams that have meaningful names (not just numbers or symbols)
            if len(team) > 2 and not team.startswith('(') and not team.isdigit():
                teams_with_full_names.append({
                    'abbreviation': team,
                    'full_name': full_name,
                    'display_name': full_name if full_name != team else f"Team {team}"
                })
        
        # Sort by display name
        teams_with_full_names.sort(key=lambda x: x['display_name'])
        
        # Get user's recent predictions
        user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()
        
        return render_template('predictions.html', teams=teams_with_full_names, predictions=user_predictions)
    except Exception as e:
        flash(f'Error loading predictions page: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        if not predictor:
            return jsonify({'success': False, 'error': 'Prediction service not available'})
        
        # Create betting opportunity
        bet = KillLineBet(
            player_name=data['player_name'],
            team=data['team'],
            opponent_team=data.get('opponent_team', 'Unknown'),
            kill_line=float(data['kill_line']),
            map=data.get('map', 'Unknown'),
            tournament=data.get('tournament', 'Unknown'),
            series_type=data.get('series_type', 'BO3'),
            maps_scope=data.get('maps_scope', 'all')
        )
        
        # Make prediction
        prediction_result = predictor.predict_kill_line(bet)
        
        # Get detailed player analysis for explanation
        player_analysis = predictor.analyze_player_history(data['player_name'], data['team'])
        
        # Generate explanation
        explanation = generate_prediction_explanation(prediction_result, player_analysis, bet)
        
        # Save prediction to database
        prediction_record = Prediction(
            user_id=current_user.id,
            player_name=data['player_name'],
            team=data['team'],
            opponent_team=data.get('opponent_team'),
            kill_line=float(data['kill_line']),
            prediction_type=prediction_result.prediction.name,
            confidence=prediction_result.confidence,
            map_name=data.get('map'),
            tournament=data.get('tournament')
        )
        
        db.session.add(prediction_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': {
                'player_name': prediction_result.player_name,
                'team': prediction_result.team,
                'kill_line': prediction_result.kill_line,
                'prediction': prediction_result.prediction.name,
                'confidence': prediction_result.confidence,
                'over_probability': prediction_result.over_probability,
                'under_probability': prediction_result.under_probability,
                'recommended_action': prediction_result.recommended_action,
                'explanation': explanation,
                'player_stats': player_analysis.get('stats', {}),
                'player_percentiles': player_analysis.get('percentiles', {}),
                'strengths': player_analysis.get('strengths', []),
                'weaknesses': player_analysis.get('weaknesses', [])
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/players/<team>')
def api_players_by_team(team):
    """Get players for a specific team"""
    try:
        players = get_players_by_team(team)
        player_list = [{'name': player[1], 'team': player[2]} for player in players]
        return jsonify({'success': True, 'players': player_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/teams')
def api_teams():
    """Get all teams"""
    try:
        teams = get_teams()
        return jsonify({'success': True, 'teams': teams})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/matches')
def matches():
    """View upcoming and past matches"""
    matches = Match.query.order_by(Match.match_date.desc()).all()
    return render_template('matches.html', matches=matches)

@app.route('/players')
def players():
    """View player statistics"""
    try:
        players_data = get_players()
        teams = get_teams()
        return render_template('players.html', players=players_data, teams=teams)
    except Exception as e:
        flash(f'Error loading players: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/admin')
@login_required
def admin():
    """Admin panel for managing the system"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('index'))
    
    # Get system stats
    stats = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_matches': Match.query.count(),
        'recent_users': User.query.order_by(User.created_at.desc()).limit(5).all()
    }
    
    return render_template('admin.html', stats=stats)

@app.route('/api/train-model', methods=['POST'])
@login_required
def api_train_model():
    """Retrain the prediction model with 5000 most recent matches and advanced features"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'})
    try:
        if not predictor:
            return jsonify({'success': False, 'error': 'Prediction service not available'})

        # Load the 5000 most recent real matches
        print("Loading 5000 most recent matches for retraining...")
        all_matches = DataLoader().load_all_matches()
        all_matches = sorted(all_matches, key=lambda m: m.get('date', ''))[-5000:]

        # Advanced feature engineering
        print("Engineering advanced features for each match...")
        historical_data = []
        for match in all_matches:
            # Extract features for each player in the match
            for player_stats in match.get('players', []):
                features = {
                    'player_name': player_stats.get('name'),
                    'team': player_stats.get('team'),
                    'opponent_team': match.get('opponent_team'),
                    'kill_line': player_stats.get('kill_line', 0),
                    'actual_kills': player_stats.get('kills', 0),
                    'map': match.get('map'),
                    'tournament': match.get('tournament'),
                    'date': match.get('date'),
                    'rounds_played': match.get('rounds_played'),
                    'rating': player_stats.get('rating'),
                    'acs': player_stats.get('acs'),
                    'kdr': player_stats.get('kdr'),
                    'assists': player_stats.get('assists'),
                    'first_kills': player_stats.get('first_kills'),
                    'first_deaths': player_stats.get('first_deaths'),
                    'headshot_percentage': player_stats.get('headshot_percentage'),
                    'clutch_success_percentage': player_stats.get('clutch_success_percentage'),
                    'agent': player_stats.get('agent'),
                    'role': player_stats.get('role'),
                    'team_rating': match.get('team_rating'),
                    'opponent_rating': match.get('opponent_rating'),
                    'recent_form': player_stats.get('recent_form'),
                    'map_winrate': player_stats.get('map_winrate'),
                    'meta_patch': match.get('meta_patch'),
                    'lan': match.get('lan'),
                    'stage': match.get('stage'),
                }
                historical_data.append(features)

        # Train the model
        print(f"Training model with {len(historical_data)} player-match records...")
        results = predictor.train_model(historical_data, save_path="models/kill_predictor.pkl")

        return jsonify({
            'success': True,
            'message': 'Model training completed',
            'accuracy': results.get('accuracy', 'N/A'),
            'label_distribution': results.get('label_distribution', {})
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        flash('Registration successful!')
        return redirect(url_for('index'))
    
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
        # Create database tables
        db.create_all()
        
        # Create admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@valorant-predictor.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username='admin', password='admin123'")
        
        # Initialize ML services
        services_initialized = initialize_services()
        if services_initialized:
            print("Application ready!")
        else:
            print("Warning: ML services failed to initialize. Some features may not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
