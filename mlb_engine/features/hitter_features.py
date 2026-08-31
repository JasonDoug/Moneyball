"""
Stage 3: Offensive & Matchup Feature Engineering
Calculates rolling wOBA, wRC+, Platoon Splits (vs LHP / vs RHP), and Pitch-Type Matchup Scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from mlb_engine.config import LEAGUE_AVERAGES

class HitterFeatureEngineer:
    """Engineers team lineup offensive metrics, platoon splits, and pitch-mix matchup scores."""

    @staticmethod
    def calculate_lineup_offensive_metrics(
        lineup_stats: pd.DataFrame,
        opponent_starter_throws: str = "R"  # 'L' or 'R'
    ) -> Dict[str, float]:
        """
        Calculates aggregate lineup metrics factoring in platoon splits (vs LHP / vs RHP).
        """
        if lineup_stats.empty:
            return {
                "lineup_rolling_woba": LEAGUE_AVERAGES["woba"],
                "lineup_rolling_wrc_plus": LEAGUE_AVERAGES["wrc_plus"],
                "lineup_platoon_woba": LEAGUE_AVERAGES["woba"],
                "lineup_platoon_wrc_plus": LEAGUE_AVERAGES["wrc_plus"],
                "top_order_woba": LEAGUE_AVERAGES["woba"]
            }

        # Platoon split columns
        split_woba_col = f"woba_vs_{opponent_starter_throws.lower()}hp"
        split_wrc_col = f"wrc_plus_vs_{opponent_starter_throws.lower()}hp"

        woba = lineup_stats["woba"].mean() if "woba" in lineup_stats else LEAGUE_AVERAGES["woba"]
        wrc_plus = lineup_stats["wrc_plus"].mean() if "wrc_plus" in lineup_stats else LEAGUE_AVERAGES["wrc_plus"]

        platoon_woba = lineup_stats[split_woba_col].mean() if split_woba_col in lineup_stats else woba
        platoon_wrc = lineup_stats[split_wrc_col].mean() if split_wrc_col in lineup_stats else wrc_plus

        # Top of order (batters 1-4) weighting
        top_order = lineup_stats.head(4)
        top_woba = top_order["woba"].mean() if "woba" in top_order else woba

        return {
            "lineup_rolling_woba": float(woba),
            "lineup_rolling_wrc_plus": float(wrc_plus),
            "lineup_platoon_woba": float(platoon_woba),
            "lineup_platoon_wrc_plus": float(platoon_wrc),
            "top_order_woba": float(top_woba)
        }

    @staticmethod
    def calculate_pitch_type_matchup_score(
        lineup_pitch_run_values: Dict[str, float],
        starter_pitch_mix: Dict[str, float]
    ) -> float:
        """
        Calculates Pitch-Type Matchup Score.
        Weighted combination of lineup's run-value per 100 pitches against pitch types (FF, SL, CH, CU, SI)
        multiplied by opponent starter's usage percentages.
        
        Score > 0 implies lineup has strong matchup advantage against pitcher's mix.
        """
        if not starter_pitch_mix or not lineup_pitch_run_values:
            return 0.0

        matchup_score = 0.0
        total_weight = 0.0

        for ptype, usage in starter_pitch_mix.items():
            if ptype in lineup_pitch_run_values:
                matchup_score += lineup_pitch_run_values[ptype] * usage
                total_weight += usage

        if total_weight > 0:
            return float(matchup_score / total_weight)
        return 0.0
