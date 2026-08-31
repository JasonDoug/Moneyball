"""
Stage 3: Pitcher Feature Engineering
Calculates rolling metrics (last 3, 5, 10 starts) and pitch-level Statcast indicators:
K-BB%, SIERA, FIP/xFIP, Velocity trends, Whiff%, Hard-Hit%, xwOBA against.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from mlb_engine.config import LEAGUE_AVERAGES

class PitcherFeatureEngineer:
    """Engineers rolling and Statcast pitch-level features for starting pitchers."""

    @staticmethod
    def calculate_rolling_pitcher_metrics(pitcher_history: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> Dict[str, float]:
        """
        Computes rolling pitch metrics across N start windows.
        Prevents lookahead bias by using strictly historical starts prior to current game.
        """
        if pitcher_history.empty:
            features = {}
            for w in windows:
                features[f"pitcher_k_bb_last_{w}"] = LEAGUE_AVERAGES["k_minus_bb"]
                features[f"pitcher_siera_last_{w}"] = LEAGUE_AVERAGES["siera"]
                features[f"pitcher_xfip_last_{w}"] = LEAGUE_AVERAGES["xfip"]
                features[f"pitcher_velo_last_{w}"] = LEAGUE_AVERAGES["fastball_velo"]
                features[f"pitcher_whiff_pct_last_{w}"] = LEAGUE_AVERAGES["whiff_pct"]
                features[f"pitcher_hard_hit_pct_last_{w}"] = LEAGUE_AVERAGES["hard_hit_pct"]
                features[f"pitcher_xwoba_last_{w}"] = LEAGUE_AVERAGES["xwoba"]
            features["pitcher_velo_trend"] = 0.0
            return features

        features = {}
        # Ensure chronological order
        df_sorted = pitcher_history.sort_values(by="date", ascending=False)

        for w in windows:
            recent = df_sorted.head(w)
            
            k_pct = recent["k_pct"].mean() if "k_pct" in recent else LEAGUE_AVERAGES["k_pct"]
            bb_pct = recent["bb_pct"].mean() if "bb_pct" in recent else LEAGUE_AVERAGES["bb_pct"]
            k_bb = (k_pct - bb_pct) if ("k_pct" in recent and "bb_pct" in recent) else recent.get("k_bb_diff", pd.Series([LEAGUE_AVERAGES["k_minus_bb"]])).mean()
            
            features[f"pitcher_k_bb_last_{w}"] = float(k_bb)
            features[f"pitcher_siera_last_{w}"] = float(recent.get("siera", pd.Series([LEAGUE_AVERAGES["siera"]])).mean())
            features[f"pitcher_xfip_last_{w}"] = float(recent.get("xfip", pd.Series([LEAGUE_AVERAGES["xfip"]])).mean())
            features[f"pitcher_velo_last_{w}"] = float(recent.get("vFA", pd.Series([LEAGUE_AVERAGES["fastball_velo"]])).mean())
            features[f"pitcher_whiff_pct_last_{w}"] = float(recent.get("whiff_pct", pd.Series([LEAGUE_AVERAGES["whiff_pct"]])).mean())
            features[f"pitcher_hard_hit_pct_last_{w}"] = float(recent.get("hard_hit_pct", pd.Series([LEAGUE_AVERAGES["hard_hit_pct"]])).mean())
            features[f"pitcher_xwoba_last_{w}"] = float(recent.get("xwoba", pd.Series([LEAGUE_AVERAGES["xwoba"]])).mean())

        # Velocity Trend: difference between last 3 starts velo vs last 10 starts velo
        v3 = features.get("pitcher_velo_last_3", LEAGUE_AVERAGES["fastball_velo"])
        v10 = features.get("pitcher_velo_last_10", LEAGUE_AVERAGES["fastball_velo"])
        features["pitcher_velo_trend"] = float(v3 - v10)

        return features

    @staticmethod
    def extract_statcast_pitcher_profile(pitch_data: pd.DataFrame, pitcher_id: int) -> Dict[str, float]:
        """Extracts pitch-level Statcast metrics (Whiff%, Hard-Hit%, xwOBA against) from raw pitch logs."""
        df_p = pitch_data[pitch_data["pitcher_id"] == pitcher_id]
        if df_p.empty:
            return {
                "statcast_whiff_rate": LEAGUE_AVERAGES["whiff_pct"],
                "statcast_hard_hit_rate": LEAGUE_AVERAGES["hard_hit_pct"],
                "statcast_xwoba_against": LEAGUE_AVERAGES["xwoba"],
                "primary_pitch_type": "FF",
                "primary_pitch_pct": 0.50
            }

        total_pitches = len(df_p)
        swings = df_p[df_p["description"].isin(["swinging_strike", "foul", "hit_into_play"])]
        whiffs = df_p[df_p["description"] == "swinging_strike"]
        whiff_rate = len(whiffs) / len(swings) if len(swings) > 0 else LEAGUE_AVERAGES["whiff_pct"]

        batted_balls = df_p[df_p["events"].notna() & (df_p["events"] != "walk") & (df_p["events"] != "strikeout")]
        hard_hits = batted_balls[batted_balls["launch_speed"] >= 95.0]
        hard_hit_rate = len(hard_hits) / len(batted_balls) if len(batted_balls) > 0 else LEAGUE_AVERAGES["hard_hit_pct"]

        xwoba_against = df_p["estimated_woba_using_speedangle"].dropna().mean()
        if pd.isna(xwoba_against):
            xwoba_against = LEAGUE_AVERAGES["xwoba"]

        pitch_counts = df_p["pitch_type"].value_counts()
        primary_pitch = pitch_counts.index[0] if not pitch_counts.empty else "FF"
        primary_pitch_pct = pitch_counts.iloc[0] / total_pitches if not pitch_counts.empty else 0.50

        return {
            "statcast_whiff_rate": float(whiff_rate),
            "statcast_hard_hit_rate": float(hard_hit_rate),
            "statcast_xwoba_against": float(xwoba_against),
            "primary_pitch_type": primary_pitch,
            "primary_pitch_pct": float(primary_pitch_pct)
        }
