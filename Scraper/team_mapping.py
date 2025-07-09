# Team name mappings from abbreviations to full names
TEAM_MAPPINGS = {
    # Major Teams
    "100T": "100 Thieves",
    "SEN": "Sentinels",
    "TSM": "Team SoloMid",
    "C9": "Cloud9",
    "TL": "Team Liquid",
    "NV": "Team Envy",
    "G2": "G2 Esports",
    "FNC": "Fnatic",
    "ACE": "Team Acend",
    "GMB": "Gambit Esports",
    "VS": "Vision Strikers",
    "KRU": "KRÜ Esports",
    "VK": "Vikings",
    "FS": "Full Sense",
    "CR": "Crazy Raccoon",
    "ZETA": "ZETA DIVISION",
    "PRX": "Paper Rex",
    "XERXIA": "XERXIA Esports",
    "LOUD": "LOUD",
    "OPTIC": "OpTic Gaming",
    "XSET": "XSET",
    "GUARD": "The Guard",
    "V1": "Version1",
    "RISE": "Rise",
    "ABX": "Absolute JUPITER",
    "LG": "Luminosity Gaming",
    "GHOST": "Ghost Gaming",
    "ANDBOX": "ANDBOX",
    "IMT": "Immortals",
    "T1": "T1",
    "GEN": "Gen.G Esports",
    "DRX": "DRX",
    "ONS": "ONS",
    "NUTURN": "NUTURN",
    "F4Q": "F4Q",
    "DAMWON": "DAMWON KIA",
    "GUILD": "Guild Esports",
    "LIQUID": "Team Liquid",
    "FPXWIN": "FunPlus Phoenix",
    "M3C": "M3 Champions",
    "NAVI": "Natus Vincere",
    "BIG": "BIG",
    "SMB": "SuperMassive Blaze",
    "OXG": "Oxygen Esports",
    "FAZE": "FaZe Clan",
    "EG": "Evil Geniuses",
    "NRG": "NRG Esports",
    "BBG": "Built By Gamers",
    "KNIGHTS": "Knights",
    "COMPLEXITY": "Complexity Gaming",
    "DIGNITAS": "Dignitas",
    "ENVY": "Team Envy",
    
    # Regional Teams
    "ARCH": "ARCH",
    "ASTA": "ASTA",
    "AWAK": "AWAK",
    "C4C": "C4C",
    "CUBE": "CUBE",
    "1337": "1337",
    "50PF": "50PF",
    "86th": "86th",
    "AKV.GC": "AKV Game Changers",
    "Abde": "Abde",
    
    # Asia-Pacific Teams
    "EDG": "Edward Gaming",
    "EDGS": "Edward Gaming",
    "EDward Gaming": "Edward Gaming",
    "EDWARD GAMING": "Edward Gaming",
    "TTN": "Titan Esports",
    "TITAN": "Titan Esports",
    "Titan": "Titan Esports",
    "PRX": "Paper Rex",
    "PAPER REX": "Paper Rex",
    "T1": "T1",
    "T1 Korea": "T1",
    "DRX": "DRX",
    "DAMWON": "DAMWON KIA",
    "DK": "DAMWON KIA",
    "GEN": "Gen.G Esports",
    "GENG": "Gen.G Esports",
    "VS": "Vision Strikers",
    "VISION STRIKERS": "Vision Strikers",
    "NUTURN": "NUTURN",
    "F4Q": "F4Q",
    "ONS": "ONS",
    "ZETA": "ZETA DIVISION",
    "ZETA DIVISION": "ZETA DIVISION",
    "CR": "Crazy Raccoon",
    "CRAZY RACCOON": "Crazy Raccoon",
    "NORTHEPTION": "NORTHEPTION",
    "NTH": "NORTHEPTION",
    "REJECT": "REJECT",
    "RJ": "REJECT",
    "FENNEL": "FENNEL",
    "FL": "FENNEL",
    "XERXIA": "XERXIA Esports",
    "XIA": "XERXIA Esports",
    "FULL SENSE": "Full Sense",
    "FS": "Full Sense",
    "BLEED": "BLEED Esports",
    "BLEED ESPORTS": "BLEED Esports",
    "BOOM": "BOOM Esports",
    "BOOM ESPORTS": "BOOM Esports",
    "ONIC": "ONIC Esports",
    "ONIC ESPORTS": "ONIC Esports",
    "RRQ": "Rex Regum Qeon",
    "REX REGUM QEON": "Rex Regum Qeon",
    "TALON": "TALON Esports",
    "TALON ESPORTS": "Talon Esports",
    "GLOBAL ESPORTS": "Global Esports",
    "GE": "Global Esports",
    "VELOCITY GAMING": "Velocity Gaming",
    "VLT": "Velocity Gaming",
    "ENIGMA GAMING": "Enigma Gaming",
    "EG": "Enigma Gaming",
    
    # Placeholder teams (numbers in parentheses)
    "(+1)": "Team +1",
    "(+2)": "Team +2",
    "(+10)": "Team +10",
    "(+11)": "Team +11",
    "(+12)": "Team +12",
    "(+13)": "Team +13",
    "(+14)": "Team +14",
    "(+16)": "Team +16",
    "(+20)": "Team +20",
}

def get_full_team_name(abbreviation):
    """
    Get the full team name from abbreviation.
    If no mapping exists, return the original abbreviation.
    """
    return TEAM_MAPPINGS.get(abbreviation, abbreviation)

def get_display_name(team_name):
    """
    Get a display-friendly team name.
    Converts abbreviations to full names where possible.
    """
    full_name = get_full_team_name(team_name)
    
    # If it's still an abbreviation or short code, make it more readable
    if len(full_name) <= 4 and full_name.isupper():
        return f"Team {full_name}"
    
    return full_name

def search_teams(teams_list, search_term):
    """
    Search teams by both abbreviation and full name.
    """
    if not search_term:
        return teams_list
    
    search_lower = search_term.lower()
    filtered_teams = []
    
    for team in teams_list:
        # Search in abbreviation
        if search_lower in team.lower():
            filtered_teams.append(team)
            continue
            
        # Search in full name
        full_name = get_full_team_name(team)
        if search_lower in full_name.lower():
            filtered_teams.append(team)
            continue
    
    return filtered_teams
