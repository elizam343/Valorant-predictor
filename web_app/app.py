import sys
import os

# Make Scraper and kill_prediction_model importable from web_app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kill_prediction_model'))

from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
import json
import sqlite3

# Load .env if present (python-dotenv is already in requirements.txt)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

from db_utils import get_players, get_teams, get_players_by_team
from prediction_engine import PredictionEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-in-prod')

# Initialise prediction engine once at startup
_engine = PredictionEngine()

# ---------------------------------------------------------------------------
# Prediction history database
# ---------------------------------------------------------------------------

_PRED_DB = os.path.join(os.path.dirname(__file__), 'predictions.db')


def _pred_conn():
    return sqlite3.connect(_PRED_DB)


def _init_predictions_db():
    with _pred_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name          TEXT NOT NULL,
                player_team          TEXT,
                opponent_team        TEXT,
                map_name             TEXT,
                series_type          TEXT,
                tournament           TEXT,
                kill_line            REAL,
                kill_line_unit       TEXT,
                kill_line_per_map    REAL,
                recommendation       TEXT,
                predicted_kills      REAL,
                predicted_user_unit  REAL,
                pct_edge             REAL,
                ci_low               REAL,
                ci_high              REAL,
                created_at           TEXT DEFAULT (datetime('now')),
                actual_kills         REAL,
                was_correct          INTEGER,
                notes                TEXT
            )
        """)


_init_predictions_db()


def _log_prediction(data: dict, prediction: dict) -> int:
    """Persist one prediction row; returns the new row id."""
    ci = prediction.get('confidence_interval_95') or [None, None]
    with _pred_conn() as conn:
        cur = conn.execute("""
            INSERT INTO predictions
              (player_name, player_team, opponent_team, map_name, series_type,
               tournament, kill_line, kill_line_unit, kill_line_per_map,
               recommendation, predicted_kills, predicted_user_unit,
               pct_edge, ci_low, ci_high)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get('player_name'),
            data.get('team') or data.get('player_team', ''),
            data.get('opponent_team', ''),
            data.get('map') or data.get('map_name', ''),
            data.get('series_type', ''),
            data.get('tournament', ''),
            data.get('kill_line'),
            data.get('kill_line_unit', 'per_map'),
            prediction.get('kill_line_per_map'),
            prediction.get('recommendation'),
            prediction.get('predicted_kills'),
            prediction.get('predicted_in_user_unit'),
            prediction.get('pct_edge'),
            ci[0], ci[1],
        ))
        return cur.lastrowid


# Provide a dummy current_user so templates that reference Flask-Login's
# current_user don't blow up (auth was removed; single-user tool).
class _AnonUser:
    is_authenticated = False
    is_admin = False
    username = ''

@app.context_processor
def _inject_user():
    return {'current_user': _AnonUser()}

# ---------------------------------------------------------------------------
# Kill line unit conversion
# ---------------------------------------------------------------------------

_AVG_ROUNDS_PER_MAP = 18.5   # typical Valorant map round count

# Expected maps played per series format (symmetric 50/50 map win rate)
_EXPECTED_MAPS = {'BO1': 1.0, 'BO3': 2.5, 'BO5': 3.5}


def _to_per_map(kill_line: float, unit: str, series_type: str = 'BO3') -> float:
    """Convert any kill-line unit to per-map kills (the model's native unit)."""
    if unit == 'per_round':
        return kill_line * _AVG_ROUNDS_PER_MAP
    if unit == 'series_total':
        maps = _EXPECTED_MAPS.get(series_type.upper(), 2.5)
        return kill_line / maps
    return kill_line   # 'per_map' — no conversion


def _from_per_map(kills_per_map: float, unit: str, series_type: str = 'BO3') -> float:
    """Convert per-map kills back to the user-chosen unit for display."""
    if unit == 'per_round':
        return kills_per_map / _AVG_ROUNDS_PER_MAP
    if unit == 'series_total':
        maps = _EXPECTED_MAPS.get(series_type.upper(), 2.5)
        return kills_per_map * maps
    return kills_per_map   # 'per_map'


_UNIT_LABELS = {
    'per_map':      'kills/map',
    'per_round':    'kills/round',
    'series_total': 'kills (series)',
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

class TeamInfo:
    """Thin wrapper so templates can access team.abbreviation and team.display_name."""
    def __init__(self, name):
        self.abbreviation = name
        self.display_name = name


def _all_teams():
    raw = get_teams()
    return [
        TeamInfo(t) for t in sorted(raw)
        if t and not str(t).startswith('(+') and not str(t).isdigit()
    ]


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    players = get_players()
    teams   = get_teams()
    stats = {
        'total_players': len(players),
        'total_teams':   len([t for t in teams if t and not str(t).startswith('(+')]),
        'last_updated':  datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    return render_template('index.html', stats=stats)


@app.route('/predictions')
def predictions():
    teams = _all_teams()
    return render_template('predictions.html', teams=teams, predictions=[])


@app.route('/players')
def players():
    all_players = get_players()
    player_dicts = [
        {
            'id':                        p[0],
            'name':                      p[1],
            'team':                      p[2],
            'rating':                    p[3],
            'average_combat_score':      p[4],
            'kill_deaths':               p[5],
            'kill_assists_survived_traded': p[6],
            'average_damage_per_round':  p[7],
            'kills_per_round':           p[8],
            'assists_per_round':         p[9],
            'first_kills_per_round':     p[10],
            'first_deaths_per_round':    p[11],
            'headshot_percentage':       p[12],
            'clutch_success_percentage': p[13],
        }
        for p in all_players
    ]
    return render_template('players.html', players=player_dicts)


@app.route('/matches')
def matches():
    return render_template('matches.html')


@app.route('/model-info')
def model_info():
    report_path = os.path.join(
        os.path.dirname(__file__), '..', 'kill_prediction_model', 'models', 'backtest_report.json'
    )
    try:
        with open(report_path) as f:
            report = json.load(f)
    except FileNotFoundError:
        report = {'error': 'Backtest report not found. Run kill_prediction_model/backtester.py first.'}
    return render_template('model_info.html', report=report)


@app.route('/api/backtest')
def api_backtest():
    report_path = os.path.join(
        os.path.dirname(__file__), '..', 'kill_prediction_model', 'models', 'backtest_report.json'
    )
    try:
        with open(report_path) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'Backtest report not found'}), 404


@app.route('/login', methods=['GET', 'POST'])
def login():
    return redirect(url_for('predictions'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    return redirect(url_for('predictions'))


@app.route('/logout')
def logout():
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route('/api/stats')
def api_stats():
    players = get_players()
    teams   = get_teams()
    return jsonify({
        'total_players': len(players),
        'total_teams':   len(teams),
        'timestamp':     datetime.now().isoformat(),
    })


@app.route('/api/players/<team_name>')
def api_players_by_team(team_name):
    try:
        rows = get_players_by_team(team_name)
        players = [
            {
                'name':              r[1],
                'team':              r[2],
                'rating':            r[3],
                'kills_per_round':   r[8],
                'average_combat_score': r[4],
            }
            for r in rows
        ]
        return jsonify({'success': True, 'players': players})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        player_name    = data.get('player_name')
        player_team    = data.get('team') or data.get('player_team', '')
        opponent_team  = data.get('opponent_team', 'Unknown')
        map_name       = data.get('map') or data.get('map_name', 'Ascent')
        series_type    = data.get('series_type', 'BO3')
        tournament     = data.get('tournament', 'Unknown Tournament')
        kill_line_raw  = data.get('kill_line')
        kill_line_unit = data.get('kill_line_unit', 'per_map')   # NEW

        if not player_name:
            return jsonify({'success': False, 'error': 'player_name is required'}), 400

        if kill_line_raw is None:
            return jsonify({'success': False, 'error': 'kill_line is required'}), 400

        kill_line_raw = float(kill_line_raw)

        # Convert user's kill line to per-map (the model's native unit)
        kill_line_per_map = _to_per_map(kill_line_raw, kill_line_unit, series_type)

        result = _engine.predict_performance(
            player_name, player_team, opponent_team,
            map_name, series_type, kill_line_per_map, tournament
        )

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400

        # Convert the model's per-map prediction back to the user's chosen unit
        predicted_in_user_unit = _from_per_map(
            result['predicted_kills'], kill_line_unit, series_type
        )
        unit_label = _UNIT_LABELS.get(kill_line_unit, 'kills/map')

        # Percentage edge (what the backtester uses for threshold decisions)
        pct_edge = abs(result['predicted_kills'] - kill_line_per_map) / kill_line_per_map * 100 \
                   if kill_line_per_map > 0 else 0

        # Betting recommendation strength based on backtest thresholds
        if pct_edge >= 20:
            edge_label = 'Strong edge (≥20%) — model historically profitable'
        elif pct_edge >= 10:
            edge_label = 'Moderate edge (10-20%) — use with caution'
        else:
            edge_label = 'Low edge (<10%) — below profitable threshold, avoid'

        recommendation = result.get('recommendation', 'UNSURE')
        prediction = {
            **result,
            # User-unit fields
            'kill_line_display':      round(kill_line_raw, 2),
            'kill_line_unit':         kill_line_unit,
            'unit_label':             unit_label,
            'predicted_in_user_unit': round(predicted_in_user_unit, 2),
            # Per-map fields (model native)
            'kill_line_per_map':      round(kill_line_per_map, 2),
            'predicted_kills_per_map': round(result['predicted_kills'], 2),
            # Betting context
            'pct_edge':               round(pct_edge, 1),
            'edge_label':             edge_label,
            # Template compatibility
            'prediction':             recommendation,
            'team':                   player_team,
            'recommended_action':     f"{'BET ' if pct_edge >= 10 else 'AVOID — '}{recommendation}",
            'over_probability':       0.65 if recommendation == 'OVER' else 0.35,
            'under_probability':      0.35 if recommendation == 'OVER' else 0.65,
            'explanation':            result.get('reasoning', ''),
            'player_stats':           {},
        }

        # Auto-save to history
        prediction['history_id'] = _log_prediction(data, prediction)

        return jsonify({'success': True, 'prediction': prediction})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# History routes
# ---------------------------------------------------------------------------

@app.route('/history')
def history():
    with _pred_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    predictions = [dict(r) for r in rows]

    # Summary stats
    settled   = [p for p in predictions if p['was_correct'] is not None]
    correct   = [p for p in settled if p['was_correct'] == 1]
    pending   = [p for p in predictions if p['was_correct'] is None]
    hit_rate  = len(correct) / len(settled) if settled else None

    # Hit rate by edge band for settled predictions
    bands = [
        ('< 10%',  lambda p: p['pct_edge'] is not None and p['pct_edge'] < 10),
        ('10-20%', lambda p: p['pct_edge'] is not None and 10 <= p['pct_edge'] < 20),
        ('≥ 20%',  lambda p: p['pct_edge'] is not None and p['pct_edge'] >= 20),
    ]
    band_stats = []
    for label, fn in bands:
        band = [p for p in settled if fn(p)]
        band_correct = sum(1 for p in band if p['was_correct'] == 1)
        band_stats.append({
            'label':    label,
            'n':        len(band),
            'correct':  band_correct,
            'hit_rate': band_correct / len(band) if band else None,
        })

    return render_template('history.html',
        predictions=predictions,
        total=len(predictions),
        pending=len(pending),
        settled=len(settled),
        correct=len(correct),
        hit_rate=hit_rate,
        band_stats=band_stats,
    )


@app.route('/api/predictions')
def api_predictions():
    with _pred_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/predictions/<int:pred_id>/result', methods=['POST'])
def api_record_result(pred_id):
    """Record actual match result for a saved prediction."""
    data = request.get_json()
    actual_kills = data.get('actual_kills')
    notes = data.get('notes', '')

    if actual_kills is None:
        return jsonify({'success': False, 'error': 'actual_kills required'}), 400

    actual_kills = float(actual_kills)

    with _pred_conn() as conn:
        row = conn.execute(
            "SELECT kill_line_per_map, recommendation FROM predictions WHERE id=?",
            (pred_id,)
        ).fetchone()

        if not row:
            return jsonify({'success': False, 'error': 'Prediction not found'}), 404

        kill_line_per_map, recommendation = row

        # Determine correctness: did the model's call match reality?
        actual_over = actual_kills > kill_line_per_map
        predicted_over = recommendation == 'OVER'
        was_correct = 1 if (actual_over == predicted_over) else 0

        conn.execute("""
            UPDATE predictions
            SET actual_kills=?, was_correct=?, notes=?
            WHERE id=?
        """, (actual_kills, was_correct, notes, pred_id))

    return jsonify({
        'success': True,
        'was_correct': bool(was_correct),
        'actual_kills': actual_kills,
        'kill_line_per_map': kill_line_per_map,
        'actual_over': actual_over,
    })


@app.route('/api/predictions/<int:pred_id>', methods=['DELETE'])
def api_delete_prediction(pred_id):
    with _pred_conn() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    return jsonify({'success': True})


@app.route('/api/update-stats', methods=['POST'])
def api_update_stats():
    """Trigger a fresh data pull from vlrggapi."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scraper'))
        import scraper_api
        import db_utils as _db
        _db.create_tables()
        players = scraper_api.fetch_all_regions()
        updated = 0
        for p in players:
            try:
                _db.upsert_player(p)
                updated += 1
            except Exception:
                pass
        return jsonify({'success': True, 'message': f'Updated {updated} player records'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Port 5000 is taken by macOS AirPlay Receiver. Default to 5001.
    # To free port 5000: System Settings → General → AirDrop & Handoff → AirPlay Receiver → Off
    parser.add_argument('--port', type=int, default=5001)
    args, _ = parser.parse_known_args()
    print(f'Starting on http://localhost:{args.port}')
    app.run(debug=True, host='0.0.0.0', port=args.port)
