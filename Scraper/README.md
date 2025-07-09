# Valorant Stats Web App

A comprehensive web application that displays Valorant players and teams from VLR.gg with up-to-date statistics, stored in a SQL database with no duplicates.

## Features

- **Real-time Data**: Fetches latest player statistics from VLR.gg API
- **Player Profiles**: Detailed individual player statistics and performance metrics
- **Team Management**: Browse teams and their rosters
- **Search & Filter**: Advanced search and filtering capabilities
- **No Duplicates**: Automatic deduplication ensures clean data
- **Responsive Design**: Works on all devices
- **Live Updates**: Update data directly from the web interface

## Project Structure

```
├── app.py                 # Main Flask web application
├── db_utils.py           # Database utilities and functions
├── scraper_api.py        # VLR.gg API scraper
├── main_app.py           # Command-line interface (legacy)
├── requirements.txt      # Python dependencies
├── vlr_players.db        # SQLite database
└── templates/            # HTML templates
    ├── base.html         # Base template with navigation
    ├── index.html        # Home page
    ├── players.html      # Players listing with search
    ├── player_profile.html # Individual player details
    ├── teams.html        # Teams listing
    └── team_players.html # Team roster view
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database
The database will be automatically created when you first run the application.

### 3. Fetch Initial Data
```bash
python scraper_api.py
```

### 4. Run the Web Application
```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Usage

### Web Interface

1. **Home Page** (`/`): Overview statistics and quick navigation
2. **Players** (`/players`): Browse all players with search and team filtering
3. **Player Profile** (`/player/<id>`): Detailed statistics for individual players
4. **Teams** (`/teams`): List all teams with player counts
5. **Team Roster** (`/team/<name>`): View all players in a specific team

### API Endpoints

- `POST /api/update-data`: Update player data from VLR.gg
- `GET /api/stats`: Get basic database statistics

### Command Line Interface

For command-line access, use the legacy interface:
```bash
python main_app.py
```

## Database Schema

### Players Table
- `id`: Primary key (auto-increment)
- `name`: Player name
- `team`: Team name
- `rating`: Overall performance rating
- `average_combat_score`: Average Combat Score
- `kill_deaths`: Kill/Death ratio
- `kill_assists_survived_traded`: KAST percentage
- `average_damage_per_round`: Average damage per round
- `kills_per_round`: Kills per round
- `assists_per_round`: Assists per round
- `first_kills_per_round`: First kills per round
- `first_deaths_per_round`: First deaths per round
- `headshot_percentage`: Headshot percentage
- `clutch_success_percentage`: Clutch success rate

**Unique Constraint**: `(name, team)` - Prevents duplicate players

## Data Source

Data is sourced from the VLR.gg API via `https://vlrggapi.vercel.app/stats`

### Supported Regions
- `na` (North America) - Default
- `eu` (Europe)
- `ap` (Asia-Pacific)
- `kr` (Korea)
- `br` (Brazil)

### Supported Timespans
- `all` (All time) - Default
- `60d` (Last 60 days)
- `30d` (Last 30 days)

## Key Features

### Automatic Deduplication
The database uses `UPSERT` operations to prevent duplicate entries. Players are uniquely identified by their name and team combination.

### Real-time Updates
Click the "Update Data" button in the web interface to fetch the latest statistics from VLR.gg.

### Responsive Design
Built with Bootstrap 5 for mobile-friendly responsive design.

### Search & Filtering
- Search players by name
- Filter players by team
- Auto-submit team filter changes

## Development

### Adding New Regions/Timespans
Modify the `fetch_players()` function in `scraper_api.py`:

```python
players = fetch_players(region="eu", timespan="30d")
```

### Customizing the Interface
Templates are located in the `templates/` directory and use Jinja2 templating with Bootstrap 5.

### Database Operations
All database operations are centralized in `db_utils.py` for easy maintenance and modification.

## Troubleshooting

### Common Issues

1. **Database locked**: Close any other applications accessing the SQLite file
2. **API timeout**: The VLR.gg API may be temporarily unavailable
3. **Missing data**: Run `python scraper_api.py` to populate initial data

### Logs
Flask runs in debug mode by default. Check the console for detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please respect VLR.gg's terms of service when using their data.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.
