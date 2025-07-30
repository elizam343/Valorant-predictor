from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

# Import our prediction service
from prediction_service import ValorantKillPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'valorant-predictor-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the prediction service
predictor = None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_name = db.Column(db.String(100), nullable=False)
    predicted_kills = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    opponent_team = db.Column(db.String(100))
    map_name = db.Column(db.String(50))
    series_type = db.Column(db.String(10))
    tournament = db.Column(db.String(100))
    kill_line = db.Column(db.Float)
    recommendation = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def initialize_services():
    """Initialize the prediction service and database"""
    global predictor
    try:
        # Initialize predictor with paths relative to web_app directory
        predictor = ValorantKillPredictor(
            model_path=None,  # Will auto-discover
            db_path="../Scraper/valorant_matches.db"
        )
        
        # Create database tables
        with app.app_context():
            db.create_all()
            
        print("✅ Services initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        return False

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    model_info = predictor.get_model_info() if predictor else {'loaded': False}
    return render_template('index.html', model_info=model_info)

@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        if not predictor or not predictor.model:
            return jsonify({
                'success': False,
                'error': 'Prediction service not available'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Extract parameters
        player_name = data.get('player_name', '').strip()
        if not player_name:
            return jsonify({
                'success': False,
                'error': 'Player name is required'
            }), 400
        
        opponent_team = data.get('opponent_team', '').strip() or None
        map_name = data.get('map_name', '').strip() or None
        series_type = data.get('series_type', 'bo3').lower()
        tournament = data.get('tournament', 'regional').lower()
        kill_line = data.get('kill_line')
        
        # Convert kill_line to float if provided
        if kill_line is not None:
            try:
                kill_line = float(kill_line)
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': 'Invalid kill line value'
                }), 400
        
        # Make prediction
        result = predictor.predict_kills(
            player_name=player_name,
            opponent_team=opponent_team,
            map_name=map_name,
            series_type=series_type,
            tournament=tournament,
            kill_line=kill_line
        )
        
        # Save prediction to database if successful
        if result.get('success') and current_user.is_authenticated:
            try:
                prediction = Prediction(
                    player_name=result['player'],
                    predicted_kills=result['predicted_kills'],
                    confidence=result['confidence'],
                    opponent_team=opponent_team,
                    map_name=map_name,
                    series_type=series_type.upper(),
                    tournament=tournament,
                    kill_line=kill_line,
                    recommendation=result.get('recommendation', {}).get('action') if result.get('recommendation') else None,
                    user_id=current_user.id
                )
                db.session.add(prediction)
                db.session.commit()
            except Exception as e:
                print(f"⚠️ Failed to save prediction: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict_series', methods=['POST'])
def api_predict_series():
    """API endpoint for making series predictions (first 2 or 3 maps)"""
    try:
        if not predictor or not predictor.model:
            return jsonify({
                'success': False,
                'error': 'Prediction service not available'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Extract parameters
        player_name = data.get('player_name', '').strip()
        if not player_name:
            return jsonify({
                'success': False,
                'error': 'Player name is required'
            }), 400
        
        opponent_team = data.get('opponent_team', '').strip() or None
        maps = data.get('maps', [])  # List of maps
        series_type = data.get('series_type', 'bo3').lower()
        tournament = data.get('tournament', 'regional').lower()
        kill_line = data.get('kill_line')
        maps_to_predict = data.get('maps_to_predict', 2)  # Default to first 2 maps
        
        # Validate maps_to_predict
        try:
            maps_to_predict = int(maps_to_predict)
            if maps_to_predict not in [2, 3]:
                return jsonify({
                    'success': False,
                    'error': 'maps_to_predict must be 2 or 3'
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid maps_to_predict value'
            }), 400
        
        # Convert kill_line to float if provided
        if kill_line is not None:
            try:
                kill_line = float(kill_line)
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': 'Invalid kill line value'
                }), 400
        
        # Validate maps list
        if maps and not isinstance(maps, list):
            return jsonify({
                'success': False,
                'error': 'Maps must be provided as a list'
            }), 400
        
        # Make series prediction
        result = predictor.predict_series_kills(
            player_name=player_name,
            opponent_team=opponent_team,
            maps=maps,
            series_type=series_type,
            tournament=tournament,
            kill_line=kill_line,
            maps_to_predict=maps_to_predict
        )
        
        # Save prediction to database if successful
        if result.get('success') and current_user.is_authenticated:
            try:
                prediction = Prediction(
                    player_name=result['player'],
                    predicted_kills=result['predicted_kills'],
                    confidence=result['confidence'],
                    opponent_team=opponent_team,
                    map_name=f"First {maps_to_predict} maps",  # Series indicator
                    series_type=series_type.upper(),
                    tournament=tournament,
                    kill_line=kill_line,
                    recommendation=result.get('recommendation', {}).get('action') if result.get('recommendation') else None,
                    user_id=current_user.id
                )
                db.session.add(prediction)
                db.session.commit()
            except Exception as e:
                print(f"⚠️ Failed to save series prediction: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Series prediction failed: {str(e)}'
        }), 500

@app.route('/api/search_players')
def api_search_players():
    """API endpoint for searching players (autocomplete)"""
    try:
        query = request.args.get('q', '').strip()
        if len(query) < 2:
            return jsonify([])
        
        if not predictor:
            return jsonify([])
        
        # Simple database query for player suggestions
        if os.path.exists(predictor.db_path):
            import sqlite3
            conn = sqlite3.connect(predictor.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT p.name 
                FROM players p 
                WHERE p.name LIKE ? 
                ORDER BY p.name 
                LIMIT 10
            """, [f"%{query}%"])
            
            players = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return jsonify(players)
        
        return jsonify([])
        
    except Exception as e:
        print(f"❌ Player search error: {e}")
        return jsonify([])

@app.route('/history')
@login_required
def prediction_history():
    """User's prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                 .order_by(Prediction.created_at.desc())\
                                 .limit(50).all()
    return render_template('history.html', predictions=predictions)

@app.route('/api/model_status')
def api_model_status():
    """Get model status and information"""
    if predictor:
        info = predictor.get_model_info()
        return jsonify(info)
    else:
        return jsonify({'loaded': False, 'error': 'Predictor not initialized'})

# Authentication routes (simplified)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered')
        else:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            login_user(user)
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
    # Initialize services before running
    if initialize_services():
        print("🚀 Starting Valorant Kill Predictor web application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize services. Please check configuration.") 