"""
Stage 1: Objective Definition & Betting Calculations
Defines Game-Level Outcomes, Player-Level Props, Odds Conversion, EV & Kelly Criterion.
"""

from typing import Dict, Tuple, Optional
import numpy as np

def american_to_decimal(american_odds: float) -> float:
    """Converts American odds (-110, +150) to Decimal odds (1.91, 2.50)."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0

def decimal_to_american(decimal_odds: float) -> float:
    """Converts Decimal odds to American odds."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    else:
        return -100.0 / (decimal_odds - 1.0)

def american_to_implied_prob(american_odds: float) -> float:
    """Converts American odds to raw implied probability (including bookmaker vig)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)

def remove_vig(odds1: float, odds2: float) -> Tuple[float, float]:
    """
    Removes vigorish from a two-way market (e.g. Moneyline or Over/Under).
    Returns (fair_prob1, fair_prob2).
    """
    p1_raw = american_to_implied_prob(odds1)
    p2_raw = american_to_implied_prob(odds2)
    total_raw = p1_raw + p2_raw
    return (p1_raw / total_raw, p2_raw / total_raw)

def calculate_expected_value(pred_prob: float, american_odds: float) -> float:
    """
    Calculates Expected Value (EV) per $1 bet.
    EV = (Win Prob * Net Profit) - (Loss Prob * Bet Amount)
    """
    dec_odds = american_to_decimal(american_odds)
    net_profit = dec_odds - 1.0
    ev = (pred_prob * net_profit) - ((1.0 - pred_prob) * 1.0)
    return ev

def kelly_criterion(pred_prob: float, american_odds: float, kelly_fraction: float = 0.25) -> float:
    """
    Calculates Kelly Criterion recommended bankroll allocation percentage.
    f* = (p * b - q) / b
    where p = win prob, q = 1 - p, b = net fractional odds (decimal_odds - 1).
    """
    dec_odds = american_to_decimal(american_odds)
    b = dec_odds - 1.0
    p = pred_prob
    q = 1.0 - p
    
    if b <= 0:
        return 0.0
        
    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0.0
    
    # Scale by fractional Kelly for risk management (e.g., quarter Kelly)
    return f_star * kelly_fraction

def evaluate_daily_lock(
    game_id: str,
    matchup: str,
    market_type: str,  # 'Moneyline', 'RunLine', 'OverUnder', 'PlayerProps'
    selection: str,    # e.g., 'Home', 'Away', 'Over 8.5', 'Under 8.5', 'Pitcher O5.5 Ks'
    pred_prob: float,
    vegas_odds: float,
    min_ev_threshold: float = 0.03,  # Minimum +3% EV
    kelly_fraction: float = 0.25
) -> Dict:
    """
    Evaluates deltas between Vegas market lines and predicted model probabilities.
    Identifies 'Daily Locks' (high positive EV betting edges).
    """
    implied_p = american_to_implied_prob(vegas_odds)
    edge = pred_prob - implied_p
    ev = calculate_expected_value(pred_prob, vegas_odds)
    kelly = kelly_criterion(pred_prob, vegas_odds, kelly_fraction=kelly_fraction)
    
    is_lock = ev >= min_ev_threshold and edge > 0.02
    
    return {
        "game_id": game_id,
        "matchup": matchup,
        "market_type": market_type,
        "selection": selection,
        "predicted_prob": round(pred_prob, 4),
        "vegas_odds": vegas_odds,
        "vegas_implied_prob": round(implied_p, 4),
        "edge": round(edge, 4),
        "expected_value": round(ev, 4),
        "kelly_bankroll_pct": round(kelly * 100, 2),
        "is_daily_lock": is_lock
    }
