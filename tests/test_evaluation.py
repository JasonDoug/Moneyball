"""
Unit Tests for Stage 5: Evaluation & Backtesting Module
"""

import pytest
import numpy as np
import pandas as pd
from mlb_engine.evaluation import (
    ModelEvaluationMetrics,
    MarketBacktester
)

def test_model_evaluation_metrics():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.1, 0.3, 0.65, 0.25])
    
    metrics = ModelEvaluationMetrics.evaluate_probabilistic_accuracy(y_true, y_prob)
    assert metrics["brier_score"] < metrics["naive_brier_score"]
    assert metrics["brier_skill_score"] > 0.0

def test_market_backtester():
    bets_df = pd.DataFrame({
        "pred_prob": [0.65, 0.70, 0.60, 0.55],
        "market_odds": [+100, -110, +120, -105],
        "actual_win": [1, 1, 0, 1]
    })
    backtester = MarketBacktester(initial_bankroll=1000.0, kelly_fraction=0.25, min_ev=0.01)
    report = backtester.backtest_bets(bets_df)
    assert report["total_bets_placed"] > 0
    assert "roi_pct" in report
