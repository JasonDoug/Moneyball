"""
Stage 4: Modeling Architecture - Two-Stage Run Expectancy Model
Uses Poisson Regression / Negative Binomial modeling to predict expected runs scored
by home team (lambda_home) and away team (lambda_away), then computes exact moneyline,
run line (-1.5 / +1.5), and Over/Under probabilities via joint Poisson / Skellam distributions.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from typing import Dict, Tuple
from sklearn.linear_model import PoissonRegressor

class TwoStageRunExpectancyModel:
    """Two-Stage Run Expectancy Model for game outcome probabilities."""

    def __init__(self, alpha: float = 1.0):
        self.home_run_model = PoissonRegressor(alpha=alpha, max_iter=1000)
        self.away_run_model = PoissonRegressor(alpha=alpha, max_iter=1000)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_home_runs: pd.Series, y_away_runs: pd.Series):
        """Trains independent Poisson regressors for home and away expected runs."""
        self.home_run_model.fit(X_train, y_home_runs)
        self.away_run_model.fit(X_train, y_away_runs)
        self.is_trained = True

    def predict_expected_runs(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts expected runs (lambda_home, lambda_away)."""
        if not self.is_trained:
            # Fallback heuristic baseline expected runs (~4.5 home, ~4.2 away)
            exp_home = np.full(len(X), 4.5)
            exp_away = np.full(len(X), 4.2)
            return exp_home, exp_away

        lambda_home = self.home_run_model.predict(X)
        lambda_away = self.away_run_model.predict(X)
        # Ensure positive expected runs
        return np.maximum(0.5, lambda_home), np.maximum(0.5, lambda_away)

    @staticmethod
    def compute_game_outcomes_from_lambdas(
        lambda_home: float,
        lambda_away: float,
        run_line: float = 1.5,
        total_line: float = 8.5,
        max_runs: int = 25
    ) -> Dict[str, float]:
        """
        Calculates exact analytical joint probabilities up to max_runs:
        - Home Moneyline Win Prob
        - Home Run Line (-1.5) Prob
        - Away Run Line (+1.5) Prob
        - Over Total Runs Prob
        - Under Total Runs Prob
        """
        h_probs = poisson.pmf(np.arange(max_runs), lambda_home)
        a_probs = poisson.pmf(np.arange(max_runs), lambda_away)

        # Joint PMF matrix: joint_pmf[h, a] = P(Home = h, Away = a)
        joint_pmf = np.outer(h_probs, a_probs)

        # 1. Moneyline
        home_win_prob = np.sum(np.tril(joint_pmf, -1))
        away_win_prob = np.sum(np.triu(joint_pmf, 1))
        tie_prob = np.sum(np.diag(joint_pmf))
        
        # In baseball, extra-innings resolve ties (assume 50/50 extra inning split)
        fair_home_win = home_win_prob + 0.5 * tie_prob
        fair_away_win = away_win_prob + 0.5 * tie_prob

        # 2. Run Line (Home -1.5 -> Home score >= Away score + 2)
        home_cover_rl = 0.0
        away_cover_rl = 0.0
        for h in range(max_runs):
            for a in range(max_runs):
                p = joint_pmf[h, a]
                if h - a > run_line:
                    home_cover_rl += p
                elif a - h > -run_line:  # Away +1.5 covers if Away > Home - 1.5
                    away_cover_rl += p

        # 3. Over / Under Total Runs
        over_prob = 0.0
        under_prob = 0.0
        for h in range(max_runs):
            for a in range(max_runs):
                total = h + a
                p = joint_pmf[h, a]
                if total > total_line:
                    over_prob += p
                elif total < total_line:
                    under_prob += p

        return {
            "expected_home_runs": round(lambda_home, 2),
            "expected_away_runs": round(lambda_away, 2),
            "expected_total_runs": round(lambda_home + lambda_away, 2),
            "home_win_prob": float(fair_home_win),
            "away_win_prob": float(fair_away_win),
            "home_run_line_cover_prob": float(home_cover_rl),
            "away_run_line_cover_prob": float(away_cover_rl),
            "over_prob": float(over_prob),
            "under_prob": float(under_prob)
        }
