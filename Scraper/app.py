from flask import Flask, render_template, request, jsonify
import db_utils
import scraper_api
import team_mapping
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Initialize database
db_utils.create_tables()

@app.route('/')
def index():
    """Home page with overview statistics"""
    players = db_utils.get_players()
    teams = db_utils.get_teams()
    
    stats = {
        'total_players': len(players),
        'total_teams': len(teams),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('index.html', stats=stats)

@app.route('/players')
def players():
    """Players page with search and filtering"""
    search = request.args.get('search', '')
    team_filter = request.args.get('team', '')
    
    all_players = db_utils.get_players()
    teams = db_utils.get_teams()
    
    # Filter players based on search and team
    filtered_players = []
    for player in all_players:
        player_dict = {
            'id': player[0],
            'name': player[1],
            'team': player[2],
            'rating': player[3],
            'average_combat_score': player[4],
            'kill_deaths': player[5],
            'kill_assists_survived_traded': player[6],
            'average_damage_per_round': player[7],
            'kills_per_round': player[8],
            'assists_per_round': player[9],
            'first_kills_per_round': player[10],
            'first_deaths_per_round': player[11],
            'headshot_percentage': player[12],
            'clutch_success_percentage': player[13]
        }
        
        # Apply filters
        if search and search.lower() not in player_dict['name'].lower():
            continue
        if team_filter and team_filter != player_dict['team']:
            continue
            
        filtered_players.append(player_dict)
    
    return render_template('players.html', 
                         players=filtered_players, 
                         teams=teams, 
                         search=search, 
                         team_filter=team_filter)

@app.route('/player/<int:player_id>')
def player_profile(player_id):
    """Individual player profile page"""
    players = db_utils.get_players()
    player = None
    
    for p in players:
        if p[0] == player_id:
            player = {
                'id': p[0],
                'name': p[1],
                'team': p[2],
                'rating': p[3],
                'average_combat_score': p[4],
                'kill_deaths': p[5],
                'kill_assists_survived_traded': p[6],
                'average_damage_per_round': p[7],
                'kills_per_round': p[8],
                'assists_per_round': p[9],
                'first_kills_per_round': p[10],
                'first_deaths_per_round': p[11],
                'headshot_percentage': p[12],
                'clutch_success_percentage': p[13]
            }
            break
    
    if not player:
        return "Player not found", 404
    
    return render_template('player_profile.html', player=player)

@app.route('/teams')
def teams():
    """Teams page with search functionality"""
    search = request.args.get('search', '')
    
    teams_list = db_utils.get_teams()
    
    # Filter out placeholder teams (teams that start with (+) or are just numbers)
    real_teams = []
    for team in teams_list:
        # Skip placeholder teams like (+1), (+2), etc.
        if team.startswith('(+') and team.endswith(')'):
            continue
        # Skip teams that are just numbers
        if team.isdigit():
            continue
        real_teams.append(team)
    
    # Apply search filter using team mapping
    if search:
        filtered_teams = team_mapping.search_teams(real_teams, search)
    else:
        filtered_teams = real_teams
    
    team_data = []
    for team in filtered_teams:
        players = db_utils.get_players_by_team(team)
        # Only include teams with at least 1 player
        if len(players) > 0:
            team_data.append({
                'name': team,
                'display_name': team_mapping.get_display_name(team),
                'player_count': len(players)
            })
    
    # Sort teams by player count (descending) then by display name
    team_data.sort(key=lambda x: (-x['player_count'], x['display_name']))
    
    return render_template('teams.html', teams=team_data, search=search)

@app.route('/team/<team_name>')
def team_players(team_name):
    """Team players page"""
    players = db_utils.get_players_by_team(team_name)
    
    team_players_data = []
    for player in players:
        team_players_data.append({
            'id': player[0],
            'name': player[1],
            'team': player[2],
            'rating': player[3],
            'average_combat_score': player[4],
            'kill_deaths': player[5],
            'kill_assists_survived_traded': player[6],
            'average_damage_per_round': player[7],
            'kills_per_round': player[8],
            'assists_per_round': player[9],
            'first_kills_per_round': player[10],
            'first_deaths_per_round': player[11],
            'headshot_percentage': player[12],
            'clutch_success_percentage': player[13]
        })
    
    return render_template('team_players.html', 
                         team_name=team_name, 
                         players=team_players_data,
                         team_mapping=team_mapping)

@app.route('/api/update-data', methods=['POST'])
def update_data():
    """API endpoint to update player data from vlr.gg"""
    try:
        # Fetch new data from all regions
        players = scraper_api.fetch_all_regions()
        
        # Update database
        updated_count = 0
        for player in players:
            try:
                db_utils.upsert_player(player)
                updated_count += 1
            except Exception as e:
                print(f"Error updating player {player.get('player', 'Unknown')}: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Updated {updated_count} player records from all regions',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for basic statistics"""
    players = db_utils.get_players()
    teams = db_utils.get_teams()
    
    return jsonify({
        'total_players': len(players),
        'total_teams': len(teams),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
