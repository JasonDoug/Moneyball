"""
Stage 4: Modeling Architecture - Direct Classification & Regression Models
Implements XGBoost, LightGBM, CatBoost, and Logistic Regression for game-level predictions.
Supports probability calibration (Platt Scaling / Isotonic Regression).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class DirectClassificationModel:
    """Direct Game Win Probability Classifier supporting multiple model backends."""

    def __init__(self, model_type: str = "xgboost", params: Optional[Dict] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self.calibrated_model = None
        self.feature_names = []

    def _build_base_model(self):
        """Instantiates selected algorithm backend."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.params.get("n_estimators", 150),
                max_depth=self.params.get("max_depth", 4),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric="logloss"
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 150),
                max_depth=self.params.get("max_depth", 4),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.8),
                random_state=42,
                verbosity=-1
            )
        elif self.model_type == "catboost":
            return CatBoostClassifier(
                iterations=self.params.get("iterations", 150),
                depth=self.params.get("depth", 4),
                learning_rate=self.params.get("learning_rate", 0.05),
                random_seed=42,
                verbose=False
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                C=self.params.get("C", 1.0),
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, calibrate: bool = True):
        """Trains base model and calibrates output probabilities using sigmoid (Platt scaling)."""
        self.feature_names = list(X_train.columns)
        base_m = self._build_base_model()

        if calibrate:
            # Uses 5-fold cross-validation calibration
            self.calibrated_model = CalibratedClassifierCV(estimator=base_m, method="sigmoid", cv=5)
            self.calibrated_model.fit(X_train, y_train)
            self.model = base_m.fit(X_train, y_train)
        else:
            self.model = base_m.fit(X_train, y_train)
            self.calibrated_model = self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated win probabilities (returns array of shape (N, 2))."""
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.calibrated_model.predict_proba(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts feature importances from base model."""
        if self.model is None:
            return pd.DataFrame()

        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imps = np.abs(self.model.coef_[0])
        else:
            imps = np.zeros(len(self.feature_names))

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": imps
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
