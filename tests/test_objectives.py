"""
Unit Tests for Stage 1: Objectives & Betting Module
"""

import pytest
from mlb_engine.objectives.betting import (
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
    remove_vig,
    calculate_expected_value,
    kelly_criterion,
    evaluate_daily_lock
)

def test_american_to_decimal():
    assert round(american_to_decimal(-110), 4) == 1.9091
    assert round(american_to_decimal(+150), 4) == 2.5000

def test_decimal_to_american():
    assert round(decimal_to_american(2.5), 1) == 150.0
    assert round(decimal_to_american(1.90909), 1) == -110.0

def test_remove_vig():
    p1, p2 = remove_vig(-110, -110)
    assert round(p1, 2) == 0.50
    assert round(p2, 2) == 0.50

def test_calculate_expected_value():
    ev = calculate_expected_value(pred_prob=0.60, american_odds=+100)
    assert round(ev, 2) == 0.20

def test_kelly_criterion():
    k = kelly_criterion(pred_prob=0.60, american_odds=+100, kelly_fraction=0.25)
    assert round(k, 3) == 0.05  # (0.60*1 - 0.40)/1 = 0.20 -> 0.20 * 0.25 = 0.05

def test_evaluate_daily_lock():
    lock = evaluate_daily_lock(
        game_id="G1", matchup="LAD vs SF", market_type="Moneyline",
        selection="LAD", pred_prob=0.65, vegas_odds=-110
    )
    assert lock["is_daily_lock"] is True
    assert lock["expected_value"] > 0
