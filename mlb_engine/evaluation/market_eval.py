"""
Stage 5: Evaluation - Market Evaluation & Kelly Criterion Backtesting
Evaluates model predictions against closing market odds (Pinnacle/sharp consensus lines)
and simulates bankroll ROI, win rate, max drawdown, and Sharpe ratio.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from mlb_engine.objectives.betting import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_expected_value,
    kelly_criterion
)

class MarketBacktester:
    """Simulates wagering strategy against real or simulated closing market lines."""

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.02,
        max_bankroll_pct_per_bet: float = 0.05
    ):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.max_bankroll_pct = max_bankroll_pct_per_bet

    def backtest_bets(
        self,
        bets_df: pd.DataFrame
    ) -> Dict:
        """
        Runs backtest on DataFrame containing:
        - pred_prob: float
        - market_odds: float (American odds e.g. -110 or +140)
        - actual_win: int (1 for win, 0 for loss)
        """
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        trades = []

        wins = 0
        losses = 0
        total_wagered = 0.0

        for idx, row in bets_df.iterrows():
            pred_p = row["pred_prob"]
            market_odds = row["market_odds"]
            actual = int(row["actual_win"])

            ev = calculate_expected_value(pred_p, market_odds)

            if ev >= self.min_ev:
                # Calculate Kelly stake percentage
                kelly_pct = kelly_criterion(pred_p, market_odds, self.kelly_fraction)
                stake_pct = min(kelly_pct, self.max_bankroll_pct)
                stake_amount = bankroll * stake_pct

                if stake_amount > 0:
                    dec_odds = american_to_decimal(market_odds)
                    total_wagered += stake_amount

                    if actual == 1:
                        profit = stake_amount * (dec_odds - 1.0)
                        bankroll += profit
                        wins += 1
                        outcome = "WIN"
                    else:
                        profit = -stake_amount
                        bankroll += profit
                        losses += 1
                        outcome = "LOSS"

                    trades.append({
                        "pred_prob": pred_p,
                        "market_odds": market_odds,
                        "ev": ev,
                        "stake": stake_amount,
                        "profit": profit,
                        "bankroll": bankroll,
                        "outcome": outcome
                    })

            bankroll_history.append(bankroll)

        # Performance summary metrics
        bankroll_arr = np.array(bankroll_history)
        net_profit = bankroll - self.initial_bankroll
        roi = (net_profit / total_wagered * 100.0) if total_wagered > 0 else 0.0
        win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0

        # Maximum Drawdown calculation
        peak = np.maximum.accumulate(bankroll_arr)
        drawdown = (bankroll_arr - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # Returns Sharpe Ratio
        if len(trades) > 1:
            returns = np.array([t["profit"] / t["stake"] for t in trades])
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(trades)) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": round(float(bankroll), 2),
            "total_net_profit": round(float(net_profit), 2),
            "total_wagered": round(float(total_wagered), 2),
            "roi_pct": round(float(roi), 2),
            "total_bets_placed": wins + losses,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(float(win_rate * 100), 2),
            "max_drawdown_pct": round(float(max_drawdown * 100), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "trades_log": trades
        }
