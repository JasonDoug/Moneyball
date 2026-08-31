"""
Unit Tests for Stage 4: Modeling Architecture Module
"""

import pytest
import numpy as np
import pandas as pd
from mlb_engine.modeling import (
    DirectClassificationModel,
    TwoStageRunExpectancyModel,
    MonteCarloGameSimulator
)

def test_direct_classification_model():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    clf = DirectClassificationModel(model_type="logistic")
    clf.train(X, y, calibrate=True)
    probs = clf.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

def test_two_stage_run_expectancy():
    outcomes = TwoStageRunExpectancyModel.compute_game_outcomes_from_lambdas(lambda_home=4.5, lambda_away=4.0)
    assert outcomes["home_win_prob"] + outcomes["away_win_prob"] == pytest.approx(1.0, abs=1e-3)
    assert outcomes["home_win_prob"] > outcomes["away_win_prob"]
    assert outcomes["over_prob"] + outcomes["under_prob"] <= 1.0

def test_monte_carlo_simulator():
    sim = MonteCarloGameSimulator(num_simulations=500, seed=42)
    res = sim.simulate_matchup(
        home_team="LAD", away_team="SF",
        home_starter_stats={"k_pct": 0.28, "bb_pct": 0.06},
        away_starter_stats={"k_pct": 0.22, "bb_pct": 0.08},
        home_lineup_stats=[{"k_pct": 0.20, "bb_pct": 0.09, "hr_rate": 0.04, "single_rate": 0.16, "double_rate": 0.05}] * 9,
        away_lineup_stats=[{"k_pct": 0.24, "bb_pct": 0.07, "hr_rate": 0.03, "single_rate": 0.14, "double_rate": 0.04}] * 9
    )
    assert 0.0 <= res["home_win_prob"] <= 1.0
    assert res["expected_home_runs"] > 0
    assert res["home_starter_expected_ks"] > 0
