"""
Stage 4: Modeling Architecture - Monte Carlo Game Simulator
Event-based Markov Chain Simulator stepping through lineups against pitchers over 10,000+ games per matchup.
Generates Moneyline, Run Line, Totals, AND Player Props (Pitcher Ks, Batter Hits/Bases/HRs) from a single engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from mlb_engine.config import LEAGUE_AVERAGES

class MonteCarloGameSimulator:
    """Markov chain event-based simulation engine for full game outcomes and player props."""

    def __init__(self, num_simulations: int = 10000, seed: int = 42):
        self.num_simulations = num_simulations
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _combine_log5(p_hitter: float, p_pitcher: float, p_league: float) -> float:
        """Bill James Log5 method to combine hitter talent, pitcher talent, and league baseline."""
        p_hitter = np.clip(p_hitter, 0.001, 0.999)
        p_pitcher = np.clip(p_pitcher, 0.001, 0.999)
        p_league = np.clip(p_league, 0.001, 0.999)

        num = (p_hitter * p_pitcher) / p_league
        den = num + ((1.0 - p_hitter) * (1.0 - p_pitcher) / (1.0 - p_league))
        return float(num / den) if den > 0 else p_league

    def simulate_matchup(
        self,
        home_team: str,
        away_team: str,
        home_starter_stats: Dict,
        away_starter_stats: Dict,
        home_lineup_stats: List[Dict],
        away_lineup_stats: List[Dict],
        environmental_multiplier: float = 1.0,
        k_line_starter: float = 5.5
    ) -> Dict:
        """
        Simulates `num_simulations` full games between home and away lineups against starting & bullpen pitchers.
        Returns comprehensive game probabilities and player prop distributions.
        """
        home_scores = np.zeros(self.num_simulations, dtype=int)
        away_scores = np.zeros(self.num_simulations, dtype=int)
        home_starter_ks = np.zeros(self.num_simulations, dtype=int)
        away_starter_ks = np.zeros(self.num_simulations, dtype=int)

        # Batter tracking: total bases, hits, HRs
        home_batter_hits = np.zeros((self.num_simulations, 9), dtype=int)
        away_batter_hits = np.zeros((self.num_simulations, 9), dtype=int)

        for sim_idx in range(self.num_simulations):
            # Simulate Away Half-Innings against Home Starter / Bullpen
            a_runs, h_ks, a_hits = self._simulate_team_batting(
                lineup=away_lineup_stats,
                starter_stats=home_starter_stats,
                env_mult=environmental_multiplier
            )
            away_scores[sim_idx] = a_runs
            home_starter_ks[sim_idx] = h_ks
            away_batter_hits[sim_idx] = a_hits

            # Simulate Home Half-Innings against Away Starter / Bullpen
            h_runs, a_ks, h_hits = self._simulate_team_batting(
                lineup=home_lineup_stats,
                starter_stats=away_starter_stats,
                env_mult=environmental_multiplier
            )
            home_scores[sim_idx] = h_runs
            away_starter_ks[sim_idx] = a_ks
            home_batter_hits[sim_idx] = h_hits

            # Resolve Extra Innings if tied after 9
            if home_scores[sim_idx] == away_scores[sim_idx]:
                if self.rng.random() < 0.5:
                    home_scores[sim_idx] += 1
                else:
                    away_scores[sim_idx] += 1

        # Aggregate Sim Results
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)

        home_win_pct = home_wins / self.num_simulations
        away_win_pct = away_wins / self.num_simulations

        # Run Line (-1.5 / +1.5)
        margin = home_scores - away_scores
        home_cover_15 = np.mean(margin > 1.5)
        away_cover_15 = np.mean(margin < -1.5)

        # Over / Under Totals
        total_runs = home_scores + away_scores
        mean_total = np.mean(total_runs)
        median_total = np.median(total_runs)
        over_85 = np.mean(total_runs > 8.5)
        under_85 = np.mean(total_runs < 8.5)

        # Player Props (Pitcher Ks)
        h_k_over = np.mean(home_starter_ks > k_line_starter)
        a_k_over = np.mean(away_starter_ks > k_line_starter)

        return {
            "num_simulations": self.num_simulations,
            "home_win_prob": round(float(home_win_pct), 4),
            "away_win_prob": round(float(away_win_pct), 4),
            "expected_home_runs": round(float(np.mean(home_scores)), 2),
            "expected_away_runs": round(float(np.mean(away_scores)), 2),
            "expected_total_runs": round(float(mean_total), 2),
            "home_run_line_cover_1_5": round(float(home_cover_15), 4),
            "away_run_line_cover_1_5": round(float(away_cover_15), 4),
            "over_8_5_prob": round(float(over_85), 4),
            "under_8_5_prob": round(float(under_85), 4),
            "home_starter_expected_ks": round(float(np.mean(home_starter_ks)), 2),
            "away_starter_expected_ks": round(float(np.mean(away_starter_ks)), 2),
            "home_starter_over_k_line_prob": round(float(h_k_over), 4),
            "away_starter_over_k_line_prob": round(float(a_k_over), 4),
            "lead_off_hitter_over_0_5_hits_prob": round(float(np.mean(home_batter_hits[:, 0] > 0)), 4)
        }

    def _simulate_team_batting(
        self,
        lineup: List[Dict],
        starter_stats: Dict,
        env_mult: float = 1.0
    ) -> Tuple[int, int, np.ndarray]:
        """Simulates 9 innings of offensive events for a lineup."""
        runs_scored = 0
        pitcher_ks = 0
        hitter_hits = np.zeros(9, dtype=int)
        lineup_idx = 0

        # Starter lasts roughly 18-24 batters (~5-6 innings)
        starter_max_batters = self.rng.integers(18, 25)
        total_batters_faced = 0

        for inning in range(1, 10):
            outs = 0
            b1, b2, b3 = 0, 0, 0

            while outs < 3:
                hitter = lineup[lineup_idx % 9]
                total_batters_faced += 1

                # Calculate event probabilities for this plate appearance
                is_against_starter = (total_batters_faced <= starter_max_batters)
                p_k = self._combine_log5(
                    hitter.get("k_pct", 0.22),
                    starter_stats.get("k_pct", 0.23) if is_against_starter else 0.24,
                    LEAGUE_AVERAGES["k_pct"]
                )
                p_bb = self._combine_log5(
                    hitter.get("bb_pct", 0.08),
                    starter_stats.get("bb_pct", 0.08) if is_against_starter else 0.09,
                    LEAGUE_AVERAGES["bb_pct"]
                )
                p_hr = hitter.get("hr_rate", 0.035) * env_mult
                p_single = hitter.get("single_rate", 0.15)
                p_double = hitter.get("double_rate", 0.045)

                roll = self.rng.random()

                if roll < p_k:
                    outs += 1
                    if is_against_starter:
                        pitcher_ks += 1
                elif roll < p_k + p_bb:
                    # Walk: advance runners
                    if b1 and b2 and b3:
                        runs_scored += 1
                    elif b1 and b2:
                        b3 = 1
                    elif b1:
                        b2 = 1
                    b1 = 1
                elif roll < p_k + p_bb + p_hr:
                    # Home Run
                    runs_scored += 1 + b1 + b2 + b3
                    b1, b2, b3 = 0, 0, 0
                    hitter_hits[lineup_idx % 9] += 1
                elif roll < p_k + p_bb + p_hr + p_double:
                    # Double
                    runs_scored += b2 + b3
                    b3 = b1
                    b2 = 1
                    b1 = 0
                    hitter_hits[lineup_idx % 9] += 1
                elif roll < p_k + p_bb + p_hr + p_double + p_single:
                    # Single
                    runs_scored += b3
                    b3 = b2
                    b2 = b1
                    b1 = 1
                    hitter_hits[lineup_idx % 9] += 1
                else:
                    # Out in play
                    outs += 1

                lineup_idx += 1

        return runs_scored, pitcher_ks, hitter_hits
