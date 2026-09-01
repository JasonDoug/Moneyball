"""
Unit Tests for Stage 3: Feature Engineering Module
"""

import pytest
import pandas as pd
from mlb_engine.features import (
    PitcherFeatureEngineer,
    HitterFeatureEngineer,
    BullpenFeatureEngineer,
    ContextFeatureEngineer
)

def test_pitcher_features():
    df_empty = pd.DataFrame()
    feats = PitcherFeatureEngineer.calculate_rolling_pitcher_metrics(df_empty)
    assert "pitcher_k_bb_last_3" in feats
    assert feats["pitcher_k_bb_last_3"] > 0

def test_hitter_features():
    df_lineup = pd.DataFrame({
        "woba": [0.320, 0.350, 0.380, 0.340],
        "wrc_plus": [105, 120, 140, 115]
    })
    feats = HitterFeatureEngineer.calculate_lineup_offensive_metrics(df_lineup, opponent_starter_throws="R")
    assert feats["lineup_rolling_woba"] > 0.300

def test_pitch_type_matchup():
    lineup_vals = {"FF": 1.2, "SL": -0.5, "CH": 0.8}
    starter_mix = {"FF": 0.50, "SL": 0.30, "CH": 0.20}
    score = HitterFeatureEngineer.calculate_pitch_type_matchup_score(lineup_vals, starter_mix)
    expected = 1.2 * 0.5 + (-0.5) * 0.3 + 0.8 * 0.2
    assert round(score, 4) == round(expected, 4)

def test_bullpen_features():
    ctx = {"bullpen_pitches_1d": 50, "bullpen_pitches_2d": 90, "bullpen_pitches_3d": 140}
    res = BullpenFeatureEngineer.calculate_bullpen_fatigue_and_quality(ctx, team_bullpen_xfip=4.00)
    assert res["bullpen_effective_xfip"] >= 4.00

def test_context_features():
    env = ContextFeatureEngineer.calculate_environmental_context(
        venue="COL",
        weather={"temperature": 85.0, "humidity": 40.0, "wind_speed": 10.0, "wind_direction": "Out to CF"}
    )
    assert env["park_factor"] == 1.18
    assert env["total_environmental_run_multiplier"] > 1.15
