"""
Global Configuration & Constants for MLB Prediction Engine
"""
import os
from pathlib import Path

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
    "COL": 1.15,  # Coors Field (High altitude, high run environment)
    "BOS": 1.06,  # Fenway Park
    "CIN": 1.05,  # Great American Ball Park
    "LAA": 1.02,
    "PHI": 1.02,
    "TEX": 1.02,
    "WSH": 1.01,
    "ATL": 1.01,
    "BAL": 1.00,
    "CHC": 1.00,
    "CWS": 0.99,
    "CLE": 0.99,
    "DET": 0.98,
    "HOU": 0.99,
    "KC":  0.99,
    "LAD": 0.98,
    "MIA": 0.95,
    "MIL": 1.01,
    "MIN": 0.99,
    "NYM": 0.96,
    "NYY": 1.02,
    "OAK": 0.94,
    "PIT": 0.97,
    "SD":  0.95,
    "SF":  0.94,  # Oracle Park (Pitcher friendly)
    "SEA": 0.95,
    "STL": 0.98,
    "TB":  0.96,
    "TOR": 1.01,
    "ARI": 1.01,
}

# Weather Adjustment Baseline Factors
WEATHER_BASELINES = {
    "temp_ref": 72.0,       # Base temperature in Fahrenheit (5 degrees ~ +1% runs)
    "humidity_ref": 50.0,   # Base humidity percentage
    "wind_ref": 0.0,        # Base wind speed in mph
}

# Default League Average Baselines for missing rolling data
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
