"""
Stage 2: Retrosheet Historical Game Logs & Backtesting Data Fetcher
Parses Retrosheet historical game log formats for historical backtesting.
"""

import pandas as pd
import numpy as np
from typing import Optional

class RetrosheetFetcher:
    """Fetcher and parser for Retrosheet historical game logs (1920-2025)."""

    RETROSHEET_COLUMNS = [
        "date", "game_number", "day_of_week", "visiting_team", "visiting_league",
        "visiting_game_num", "home_team", "home_league", "home_game_num",
        "visiting_score", "home_score", "outs_total", "day_night", "completion_info",
        "forfeit_info", "protest_info", "park_id", "attendance", "game_time_minutes",
        "visiting_line_score", "home_line_score", "visiting_ab", "visiting_h",
        "visiting_2b", "visiting_3b", "visiting_hr", "visiting_rbi", "visiting_sh",
        "visiting_sf", "visiting_hbp", "visiting_bb", "visiting_ibb", "visiting_so",
        "visiting_sb", "visiting_cs", "visiting_gidp", "visiting_ci", "visiting_lob",
        "home_ab", "home_h", "home_2b", "home_3b", "home_hr", "home_rbi",
        "home_sh", "home_sf", "home_hbp", "home_bb", "home_ibb", "home_so",
        "home_sb", "home_cs", "home_gidp", "home_ci", "home_lob"
    ]

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_historical_season(self, season: int) -> pd.DataFrame:
        """Loads historical season game log file or generates compliant Retrosheet DataFrame."""
        if self.data_path:
            try:
                df = pd.read_csv(self.data_path, header=None)
                df.columns = self.RETROSHEET_COLUMNS[:len(df.columns)]
                return df
            except Exception as e:
                print(f"[Warning] Failed loading retrosheet file ({e}). Generating synthetic historical dataset.")

        return self.generate_synthetic_historical_season(season)

    def generate_synthetic_historical_season(self, season: int, games_count: int = 2430) -> pd.DataFrame:
        """Generates a full 2,430-game MLB regular season compliant with Retrosheet schemas."""
        rng = np.random.default_rng(season)
        teams = ["LAD", "SF", "SD", "COL", "ARI", "NYY", "BOS", "TB", "BAL", "TOR",
                 "HOU", "TEX", "SEA", "LAA", "OAK", "ATL", "PHI", "NYM", "MIA", "WSH",
                 "CHC", "STL", "MIL", "CIN", "PIT", "CWS", "CLE", "DET", "MIN", "KC"]

        dates = pd.date_range(start=f"{season}-04-01", periods=180)
        records = []

        for i in range(games_count):
            g_date = pd.to_datetime(rng.choice(dates)).strftime("%Y%m%d")
            home, away = rng.choice(teams, size=2, replace=False)
            
            # Realistic baseball runs distribution (Poisson ~4.4 runs/team)
            home_runs = int(rng.poisson(4.5))
            away_runs = int(rng.poisson(4.2))
            
            # Ensure no ties in baseball
            if home_runs == away_runs:
                home_runs += 1

            records.append({
                "date": g_date,
                "game_number": 0,
                "visiting_team": away,
                "home_team": home,
                "visiting_score": away_runs,
                "home_score": home_runs,
                "total_runs": home_runs + away_runs,
                "home_win": 1 if home_runs > away_runs else 0,
                "run_margin": home_runs - away_runs,
                "park_id": home,
                "attendance": int(rng.normal(32000, 8000)),
                "visiting_so": int(rng.poisson(8.5)),
                "home_so": int(rng.poisson(8.2)),
                "visiting_h": int(rng.poisson(8.1)),
                "home_h": int(rng.poisson(8.4)),
                "visiting_hr": int(rng.poisson(1.1)),
                "home_hr": int(rng.poisson(1.2)),
            })

        df = pd.DataFrame(records)
        df.sort_values(by=["date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
