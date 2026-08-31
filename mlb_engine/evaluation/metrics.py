"""
Stage 5: Evaluation - Performance Metrics & Reliability Calibration Diagrams
Computes Brier Score, Log-Loss, Reliability Diagram data, and Expected Calibration Error (ECE).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import log_loss, brier_score_loss

class ModelEvaluationMetrics:
    """Calculates probabilistic metrics and calibration statistics."""

    @staticmethod
    def evaluate_probabilistic_accuracy(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes Log-Loss, Brier Score, and Baseline comparison against 50/50 naive model.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        # Clip probabilities to avoid log(0) undefined behavior
        y_prob_clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

        ll = log_loss(y_true, y_prob_clipped)
        bs = brier_score_loss(y_true, y_prob)

        # Baseline comparison (naive 0.50 guess)
        naive_prob = np.full_like(y_prob, 0.50)
        naive_ll = log_loss(y_true, naive_prob)
        naive_bs = brier_score_loss(y_true, naive_prob)

        # Brier Skill Score (BSS) = 1 - (BS_model / BS_baseline)
        brier_skill_score = 1.0 - (bs / naive_bs)

        return {
            "log_loss": round(float(ll), 4),
            "brier_score": round(float(bs), 4),
            "naive_log_loss": round(float(naive_ll), 4),
            "naive_brier_score": round(float(naive_bs), 4),
            "brier_skill_score": round(float(brier_skill_score), 4)
        }

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Computes calibration bins for Reliability Diagram and calculates Expected Calibration Error (ECE).
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(y_prob, bins) - 1

        prob_pred_list = []
        prob_true_list = []
        bin_counts = []
        ece = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            mask = bin_assignments == i
            n_in_bin = np.sum(mask)
            bin_counts.append(int(n_in_bin))

            if n_in_bin > 0:
                avg_pred = float(np.mean(y_prob[mask]))
                avg_true = float(np.mean(y_true[mask]))
                prob_pred_list.append(round(avg_pred, 4))
                prob_true_list.append(round(avg_true, 4))
                ece += (n_in_bin / total_samples) * abs(avg_pred - avg_true)
            else:
                prob_pred_list.append(round((bins[i] + bins[i+1])/2, 4))
                prob_true_list.append(0.0)

        return {
            "bins": bins.tolist(),
            "mean_predicted_probability": prob_pred_list,
            "fraction_of_positives": prob_true_list,
            "bin_counts": bin_counts,
            "expected_calibration_error": round(float(ece), 4)
        }
