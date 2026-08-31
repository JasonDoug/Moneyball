"""
Stage 3: Bullpen & Fatigue Feature Engineering
Calculates rolling bullpen workload (pitches thrown in last 1, 2, 3 days), bullpen xFIP,
and composite bullpen fatigue indices.
"""

import pandas as pd
import numpy as np
from typing import Dict
from mlb_engine.config import LEAGUE_AVERAGES

class BullpenFeatureEngineer:
    """Engineers bullpen fatigue and quality metrics."""

    @staticmethod
    def calculate_bullpen_fatigue_and_quality(game_context: Dict, team_bullpen_xfip: float = 4.10) -> Dict[str, float]:
        """
        Calculates bullpen usage (1d, 2d, 3d pitch counts) and composite fatigue penalty.
        """
        p1 = game_context.get("bullpen_pitches_1d", 35)
        p2 = game_context.get("bullpen_pitches_2d", 85)
        p3 = game_context.get("bullpen_pitches_3d", 130)

        # Baseline expected bullpen pitches over 3 days is ~120 pitches.
        # Workload ratio: > 1.2 indicates heavily taxed / fatigued bullpen
        workload_ratio = (p1 * 1.5 + p2 * 1.0 + p3 * 0.5) / 120.0

        # Fatigue penalty scales bullpen xFIP up by up to +0.50 runs when heavily fatigued
        fatigue_penalty = max(0.0, (workload_ratio - 1.0) * 0.35)
        effective_bullpen_xfip = team_bullpen_xfip + fatigue_penalty

        return {
            "bullpen_pitches_1d": float(p1),
            "bullpen_pitches_2d": float(p2),
            "bullpen_pitches_3d": float(p3),
            "bullpen_workload_ratio": float(workload_ratio),
            "bullpen_xfip_raw": float(team_bullpen_xfip),
            "bullpen_effective_xfip": float(effective_bullpen_xfip),
            "bullpen_fatigue_penalty": float(fatigue_penalty)
        }
