"""
Global Configuration & Constants for MLB Prediction Engine
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "mlb_data.sqlite"
PARQUET_DIR = DATA_DIR / "parquet"
PARQUET_DIR.mkdir(exist_ok=True)

# Random Seed for Reproducibility
SEED = 42

# Standard MLB Park Factors (Run Environment Scaling Factor, 1.00 = Neutral)
PARK_FACTORS = {
    "COL": 1.18, "Colorado Rockies": 1.18, "Rockies": 1.18,
    "BOS": 1.06, "Boston Red Sox": 1.06, "Red Sox": 1.06,
    "CIN": 1.05, "Cincinnati Reds": 1.05, "Reds": 1.05,
    "LAA": 1.02, "Los Angeles Angels": 1.02, "Angels": 1.02,
    "PHI": 1.02, "Philadelphia Phillies": 1.02, "Phillies": 1.02,
    "TEX": 1.02, "Texas Rangers": 1.02, "Rangers": 1.02,
    "WSH": 1.01, "Washington Nationals": 1.01, "Nationals": 1.01,
    "ATL": 1.01, "Atlanta Braves": 1.01, "Braves": 1.01,
    "BAL": 1.00, "Baltimore Orioles": 1.00, "Orioles": 1.00,
    "CHC": 1.00, "Chicago Cubs": 1.00, "Cubs": 1.00,
    "CWS": 0.99, "Chicago White Sox": 0.99, "White Sox": 0.99,
    "CLE": 0.99, "Cleveland Guardians": 0.99, "Guardians": 0.99,
    "DET": 0.98, "Detroit Tigers": 0.98, "Tigers": 0.98,
    "HOU": 0.99, "Houston Astros": 0.99, "Astros": 0.99,
    "KC":  0.99, "Kansas City Royals": 0.99, "Royals": 0.99,
    "LAD": 0.98, "Los Angeles Dodgers": 0.98, "Dodgers": 0.98,
    "MIA": 0.95, "Miami Marlins": 0.95, "Marlins": 0.95,
    "MIL": 1.01, "Milwaukee Brewers": 1.01, "Brewers": 1.01,
    "MIN": 0.99, "Minnesota Twins": 0.99, "Twins": 0.99,
    "NYM": 0.96, "New York Mets": 0.96, "Mets": 0.96,
    "NYY": 1.02, "New York Yankees": 1.02, "Yankees": 1.02,
    "OAK": 0.94, "Athletics": 0.94, "ATH": 0.94,
    "PIT": 0.97, "Pittsburgh Pirates": 0.97, "Pirates": 0.97,
    "SD":  0.95, "San Diego Padres": 0.95, "Padres": 0.95,
    "SF":  0.94, "San Francisco Giants": 0.94, "Giants": 0.94,
    "SEA": 0.95, "Seattle Mariners": 0.95, "Mariners": 0.95,
    "STL": 0.98, "St. Louis Cardinals": 0.98, "Cardinals": 0.98,
    "TB":  0.96, "Tampa Bay Rays": 0.96, "Rays": 0.96,
    "TOR": 1.01, "Toronto Blue Jays": 1.01, "Blue Jays": 1.01,
    "ARI": 1.01, "Arizona Diamondbacks": 1.01, "Diamondbacks": 1.01,
}

# Weather Adjustment Baseline Factors
WEATHER_BASELINES = {
    "temp_ref": 72.0,
    "humidity_ref": 50.0,
    "wind_ref": 0.0,
}

# Default League Average Baselines
LEAGUE_AVERAGES = {
    "k_pct": 0.225,
    "bb_pct": 0.082,
    "k_minus_bb": 0.143,
    "siera": 4.10,
    "fip": 4.15,
    "xfip": 4.10,
    "whiff_pct": 0.250,
    "hard_hit_pct": 0.380,
    "xwoba": 0.315,
    "woba": 0.315,
    "wrc_plus": 100.0,
    "bullpen_xfip": 4.10,
    "fastball_velo": 93.8,
}

# Master Team Tier Ratings with full names, short names, and 3-letter codes
TEAM_RATINGS = {
    # Dodgers
    "LAD": {"woba": 0.345, "wrc_plus": 122, "siera": 3.45, "k_bb": 0.21, "bp_xfip": 3.55},
    "Los Angeles Dodgers": {"woba": 0.345, "wrc_plus": 122, "siera": 3.45, "k_bb": 0.21, "bp_xfip": 3.55},
    # Yankees
    "NYY": {"woba": 0.342, "wrc_plus": 120, "siera": 3.55, "k_bb": 0.20, "bp_xfip": 3.65},
    "New York Yankees": {"woba": 0.342, "wrc_plus": 120, "siera": 3.55, "k_bb": 0.20, "bp_xfip": 3.65},
    # Braves
    "ATL": {"woba": 0.338, "wrc_plus": 116, "siera": 3.50, "k_bb": 0.20, "bp_xfip": 3.60},
    "Atlanta Braves": {"woba": 0.338, "wrc_plus": 116, "siera": 3.50, "k_bb": 0.20, "bp_xfip": 3.60},
    # Phillies
    "PHI": {"woba": 0.335, "wrc_plus": 114, "siera": 3.45, "k_bb": 0.22, "bp_xfip": 3.50},
    "Philadelphia Phillies": {"woba": 0.335, "wrc_plus": 114, "siera": 3.45, "k_bb": 0.22, "bp_xfip": 3.50},
    # Orioles
    "BAL": {"woba": 0.332, "wrc_plus": 112, "siera": 3.60, "k_bb": 0.19, "bp_xfip": 3.70},
    "Baltimore Orioles": {"woba": 0.332, "wrc_plus": 112, "siera": 3.60, "k_bb": 0.19, "bp_xfip": 3.70},
    # Astros
    "HOU": {"woba": 0.330, "wrc_plus": 110, "siera": 3.65, "k_bb": 0.18, "bp_xfip": 3.75},
    "Houston Astros": {"woba": 0.330, "wrc_plus": 110, "siera": 3.65, "k_bb": 0.18, "bp_xfip": 3.75},
    # Red Sox
    "BOS": {"woba": 0.328, "wrc_plus": 108, "siera": 3.90, "k_bb": 0.16, "bp_xfip": 3.95},
    "Boston Red Sox": {"woba": 0.328, "wrc_plus": 108, "siera": 3.90, "k_bb": 0.16, "bp_xfip": 3.95},
    # Diamondbacks
    "ARI": {"woba": 0.328, "wrc_plus": 108, "siera": 3.95, "k_bb": 0.15, "bp_xfip": 4.00},
    "Arizona Diamondbacks": {"woba": 0.328, "wrc_plus": 108, "siera": 3.95, "k_bb": 0.15, "bp_xfip": 4.00},
    # Mariners
    "SEA": {"woba": 0.308, "wrc_plus": 98, "siera": 3.30, "k_bb": 0.23, "bp_xfip": 3.40},
    "Seattle Mariners": {"woba": 0.308, "wrc_plus": 98, "siera": 3.30, "k_bb": 0.23, "bp_xfip": 3.40},
    # Padres
    "SD": {"woba": 0.325, "wrc_plus": 106, "siera": 3.75, "k_bb": 0.17, "bp_xfip": 3.80},
    "San Diego Padres": {"woba": 0.325, "wrc_plus": 106, "siera": 3.75, "k_bb": 0.17, "bp_xfip": 3.80},
    # Brewers
    "MIL": {"woba": 0.322, "wrc_plus": 104, "siera": 3.70, "k_bb": 0.18, "bp_xfip": 3.65},
    "Milwaukee Brewers": {"woba": 0.322, "wrc_plus": 104, "siera": 3.70, "k_bb": 0.18, "bp_xfip": 3.65},
    # Cubs
    "CHC": {"woba": 0.320, "wrc_plus": 102, "siera": 3.85, "k_bb": 0.16, "bp_xfip": 3.90},
    "Chicago Cubs": {"woba": 0.320, "wrc_plus": 102, "siera": 3.85, "k_bb": 0.16, "bp_xfip": 3.90},
    # Twins
    "MIN": {"woba": 0.318, "wrc_plus": 101, "siera": 3.80, "k_bb": 0.17, "bp_xfip": 3.85},
    "Minnesota Twins": {"woba": 0.318, "wrc_plus": 101, "siera": 3.80, "k_bb": 0.17, "bp_xfip": 3.85},
    # Guardians
    "CLE": {"woba": 0.315, "wrc_plus": 100, "siera": 3.75, "k_bb": 0.18, "bp_xfip": 3.50},
    "Cleveland Guardians": {"woba": 0.315, "wrc_plus": 100, "siera": 3.75, "k_bb": 0.18, "bp_xfip": 3.50},
    # Mets
    "NYM": {"woba": 0.322, "wrc_plus": 105, "siera": 3.90, "k_bb": 0.16, "bp_xfip": 3.95},
    "New York Mets": {"woba": 0.322, "wrc_plus": 105, "siera": 3.90, "k_bb": 0.16, "bp_xfip": 3.95},
    # Tigers
    "DET": {"woba": 0.312, "wrc_plus": 98, "siera": 3.60, "k_bb": 0.19, "bp_xfip": 3.70},
    "Detroit Tigers": {"woba": 0.312, "wrc_plus": 98, "siera": 3.60, "k_bb": 0.19, "bp_xfip": 3.70},
    # Rays
    "TB": {"woba": 0.310, "wrc_plus": 97, "siera": 3.80, "k_bb": 0.17, "bp_xfip": 3.75},
    "Tampa Bay Rays": {"woba": 0.310, "wrc_plus": 97, "siera": 3.80, "k_bb": 0.17, "bp_xfip": 3.75},
    # Giants
    "SF": {"woba": 0.310, "wrc_plus": 97, "siera": 3.70, "k_bb": 0.18, "bp_xfip": 3.80},
    "San Francisco Giants": {"woba": 0.310, "wrc_plus": 97, "siera": 3.70, "k_bb": 0.18, "bp_xfip": 3.80},
    # Blue Jays
    "TOR": {"woba": 0.312, "wrc_plus": 98, "siera": 4.10, "k_bb": 0.14, "bp_xfip": 4.15},
    "Toronto Blue Jays": {"woba": 0.312, "wrc_plus": 98, "siera": 4.10, "k_bb": 0.14, "bp_xfip": 4.15},
    # Cardinals
    "STL": {"woba": 0.310, "wrc_plus": 96, "siera": 4.20, "k_bb": 0.13, "bp_xfip": 4.25},
    "St. Louis Cardinals": {"woba": 0.310, "wrc_plus": 96, "siera": 4.20, "k_bb": 0.13, "bp_xfip": 4.25},
    # Reds
    "CIN": {"woba": 0.308, "wrc_plus": 95, "siera": 4.25, "k_bb": 0.13, "bp_xfip": 4.30},
    "Cincinnati Reds": {"woba": 0.308, "wrc_plus": 95, "siera": 4.25, "k_bb": 0.13, "bp_xfip": 4.30},
    # Royals
    "KC": {"woba": 0.315, "wrc_plus": 100, "siera": 3.85, "k_bb": 0.16, "bp_xfip": 3.90},
    "Kansas City Royals": {"woba": 0.315, "wrc_plus": 100, "siera": 3.85, "k_bb": 0.16, "bp_xfip": 3.90},
    # Pirates
    "PIT": {"woba": 0.302, "wrc_plus": 92, "siera": 3.90, "k_bb": 0.17, "bp_xfip": 4.05},
    "Pittsburgh Pirates": {"woba": 0.302, "wrc_plus": 92, "siera": 3.90, "k_bb": 0.17, "bp_xfip": 4.05},
    # Nationals
    "WSH": {"woba": 0.300, "wrc_plus": 90, "siera": 4.45, "k_bb": 0.11, "bp_xfip": 4.40},
    "Washington Nationals": {"woba": 0.300, "wrc_plus": 90, "siera": 4.45, "k_bb": 0.11, "bp_xfip": 4.40},
    # Marlins
    "MIA": {"woba": 0.295, "wrc_plus": 87, "siera": 4.50, "k_bb": 0.10, "bp_xfip": 4.50},
    "Miami Marlins": {"woba": 0.295, "wrc_plus": 87, "siera": 4.50, "k_bb": 0.10, "bp_xfip": 4.50},
    # Rangers
    "TEX": {"woba": 0.315, "wrc_plus": 100, "siera": 4.05, "k_bb": 0.15, "bp_xfip": 4.10},
    "Texas Rangers": {"woba": 0.315, "wrc_plus": 100, "siera": 4.05, "k_bb": 0.15, "bp_xfip": 4.10},
    # Rockies
    "COL": {"woba": 0.305, "wrc_plus": 88, "siera": 4.90, "k_bb": 0.08, "bp_xfip": 4.85},
    "Colorado Rockies": {"woba": 0.305, "wrc_plus": 88, "siera": 4.90, "k_bb": 0.08, "bp_xfip": 4.85},
    # White Sox
    "CWS": {"woba": 0.282, "wrc_plus": 78, "siera": 4.85, "k_bb": 0.08, "bp_xfip": 4.80},
    "Chicago White Sox": {"woba": 0.282, "wrc_plus": 78, "siera": 4.85, "k_bb": 0.08, "bp_xfip": 4.80},
    # Athletics
    "OAK": {"woba": 0.305, "wrc_plus": 92, "siera": 4.40, "k_bb": 0.12, "bp_xfip": 4.35},
    "ATH": {"woba": 0.305, "wrc_plus": 92, "siera": 4.40, "k_bb": 0.12, "bp_xfip": 4.35},
    "Athletics": {"woba": 0.305, "wrc_plus": 92, "siera": 4.40, "k_bb": 0.12, "bp_xfip": 4.35},
    # Angels
    "LAA": {"woba": 0.302, "wrc_plus": 91, "siera": 4.55, "k_bb": 0.10, "bp_xfip": 4.60},
    "Los Angeles Angels": {"woba": 0.302, "wrc_plus": 91, "siera": 4.55, "k_bb": 0.10, "bp_xfip": 4.60},
}

# Individual Starting Pitcher Ratings Lookup
STARTER_RATINGS = {
    "Paul Skenes": {"siera": 2.65, "k_bb": 0.25, "velo": 98.8},
    "Tarik Skubal": {"siera": 2.60, "k_bb": 0.26, "velo": 97.2},
    "Zack Wheeler": {"siera": 2.80, "k_bb": 0.23, "velo": 96.5},
    "Chris Sale": {"siera": 2.85, "k_bb": 0.25, "velo": 94.8},
    "Jacob deGrom": {"siera": 2.50, "k_bb": 0.28, "velo": 98.5},
    "Framber Valdez": {"siera": 3.20, "k_bb": 0.18, "velo": 94.0},
    "Logan Gilbert": {"siera": 3.25, "k_bb": 0.20, "velo": 96.0},
    "George Kirby": {"siera": 3.15, "k_bb": 0.22, "velo": 96.2},
    "Blake Snell": {"siera": 3.20, "k_bb": 0.20, "velo": 95.5},
    "Corbin Burnes": {"siera": 3.10, "k_bb": 0.21, "velo": 96.0},
    "Kevin Gausman": {"siera": 3.40, "k_bb": 0.19, "velo": 94.5},
    "Shane Bieber": {"siera": 3.45, "k_bb": 0.18, "velo": 92.5},
    "Freddy Peralta": {"siera": 3.35, "k_bb": 0.20, "velo": 94.5},
    "Zac Gallen": {"siera": 3.50, "k_bb": 0.17, "velo": 93.8},
    "Luis Castillo": {"siera": 3.40, "k_bb": 0.19, "velo": 96.5},
    "Carlos Rodón": {"siera": 3.65, "k_bb": 0.17, "velo": 95.0},
    "Seth Lugo": {"siera": 3.60, "k_bb": 0.16, "velo": 92.8},
    "Jack Flaherty": {"siera": 3.40, "k_bb": 0.21, "velo": 93.5},
}

def get_team_rating(team_identifier: str) -> Dict[str, float]:
    """Helper to safely retrieve team rating dict regardless of full name or abbreviation."""
    if not team_identifier:
        return {"woba": 0.315, "wrc_plus": 100, "siera": 4.10, "k_bb": 0.15, "bp_xfip": 4.10}
    
    # Direct lookup
    if team_identifier in TEAM_RATINGS:
        return TEAM_RATINGS[team_identifier]
    
    # Partial matching
    for key, val in TEAM_RATINGS.items():
        if key.lower() in team_identifier.lower() or team_identifier.lower() in key.lower():
            return val
            
    return {"woba": 0.315, "wrc_plus": 100, "siera": 4.10, "k_bb": 0.15, "bp_xfip": 4.10}

def get_starter_rating(starter_name: str, team_name: str) -> Dict[str, float]:
    """Helper to retrieve individual starter stats if available, else fallback to team pitching staff rating."""
    if starter_name and starter_name in STARTER_RATINGS:
        s_info = STARTER_RATINGS[starter_name]
        return {
            "siera": s_info["siera"],
            "k_bb": s_info["k_bb"],
            "velo": s_info["velo"]
        }
    
    t_info = get_team_rating(team_name)
    return {
        "siera": t_info["siera"],
        "k_bb": t_info["k_bb"],
        "velo": 93.8
    }
