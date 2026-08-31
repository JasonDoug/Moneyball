"""
Stage 2: PyBaseball Data Fetcher
Retrieves Statcast pitch-level data (Baseball Savant), FanGraphs metrics, and Baseball-Reference logs.
Includes robust error handling and fallback generators for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

class PyBaseballFetcher:
    """Wrapper around pybaseball for Statcast, FanGraphs, and BRef data ingestion."""

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self._pybaseball_available = False
        if not offline_mode:
            try:
                import pybaseball
                self._pybaseball_available = True
            except ImportError:
                self._pybaseball_available = False

    def fetch_statcast_pitch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches pitch-level Statcast data from Baseball Savant via pybaseball.
        Returns cleaned pitch-level DataFrame.
        """
        if self._pybaseball_available:
            try:
                from pybaseball import statcast
                df = statcast(start_date=start_date, end_date=end_date)
                if not df.empty:
                    df['is_strikeout'] = (df['events'] == 'strikeout').astype(int)
                    df['is_walk'] = (df['events'] == 'walk').astype(int)
                    df['is_swinging_strike'] = (df['description'] == 'swinging_strike').astype(int)
                    return df
            except Exception as e:
                print(f"[Warning] PyBaseball Statcast fetch error ({e}). Using synthetic dataset.")

        return self.generate_synthetic_statcast_data(start_date, end_date)

    def fetch_fangraphs_pitcher_leaderboard(self, season: int) -> pd.DataFrame:
        """Fetches seasonal FanGraphs pitching metrics (K-BB%, SIERA, FIP, xFIP)."""
        if self._pybaseball_available:
            try:
                from pybaseball import pitching_stats
                df = pitching_stats(season)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"[Warning] PyBaseball FanGraphs fetch error ({e}).")

        return self.generate_synthetic_fangraphs_pitchers(season)

    def generate_synthetic_statcast_data(self, start_date: str, end_date: str, num_pitches: int = 500) -> pd.DataFrame:
        """Generates realistic synthetic pitch-level Statcast data for modeling & testing."""
        dates = pd.date_range(start=start_date, end=end_date, periods=num_pitches)
        pitch_types = ["FF", "SL", "CH", "CU", "SI"]
        events = ["single", "strikeout", "field_out", "walk", "home_run", "double", "strikeout", "field_out"]
        descriptions = ["swinging_strike", "called_strike", "ball", "hit_into_play", "foul"]

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "pitch_id": [f"P_{i}" for i in range(num_pitches)],
            "game_id": [f"G_{i % 50}" for i in range(num_pitches)],
            "game_date": dates.strftime("%Y-%m-%d"),
            "pitcher_id": rng.choice([605151, 669203, 543037, 656302, 608337], size=num_pitches),
            "batter_id": rng.choice([592450, 665742, 660670, 545361, 670541], size=num_pitches),
            "pitch_type": rng.choice(pitch_types, size=num_pitches),
            "release_speed": rng.normal(94.0, 3.5, size=num_pitches).round(1),
            "events": rng.choice(events, size=num_pitches),
            "description": rng.choice(descriptions, size=num_pitches),
            "estimated_woba_using_speedangle": rng.uniform(0.150, 0.600, size=num_pitches).round(3),
            "launch_speed": rng.normal(88.5, 12.0, size=num_pitches).round(1),
            "launch_angle": rng.normal(12.0, 18.0, size=num_pitches).round(1),
        })
        df['is_strikeout'] = (df['events'] == 'strikeout').astype(int)
        df['is_walk'] = (df['events'] == 'walk').astype(int)
        df['is_swinging_strike'] = (df['description'] == 'swinging_strike').astype(int)
        return df

    def generate_synthetic_fangraphs_pitchers(self, season: int, num_pitchers: int = 30) -> pd.DataFrame:
        """Generates realistic synthetic FanGraphs pitcher leaderboards."""
        rng = np.random.default_rng(42)
        pitcher_names = [f"Pitcher_{i}" for i in range(num_pitchers)]
        ids = [600000 + i for i in range(num_pitchers)]
        
        return pd.DataFrame({
            "IDfg": ids,
            "Name": pitcher_names,
            "Season": season,
            "K%": rng.uniform(0.16, 0.35, size=num_pitchers).round(3),
            "BB%": rng.uniform(0.04, 0.12, size=num_pitchers).round(3),
            "K-BB%": rng.uniform(0.08, 0.28, size=num_pitchers).round(3),
            "SIERA": rng.uniform(2.80, 5.20, size=num_pitchers).round(2),
            "FIP": rng.uniform(2.90, 5.10, size=num_pitchers).round(2),
            "xFIP": rng.uniform(3.00, 5.00, size=num_pitchers).round(2),
            "vFA (pi)": rng.uniform(90.5, 99.0, size=num_pitchers).round(1),
            "Whiff%": rng.uniform(0.18, 0.36, size=num_pitchers).round(3),
            "HardHit%": rng.uniform(0.30, 0.48, size=num_pitchers).round(3),
        })
