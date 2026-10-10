"""
CARE-PULSE AI — Probability Calibration Service
Assesses and optimizes probabilistic calibration using Platt scaling (sigmoid) and
Isotonic regression. Generates reliability diagram coordinates, ECE, and MCE.
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class CalibrationService:
    """Manages probability calibration assessment and post-hoc adjustment."""

    def compute_reliability_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Generate coordinates for reliability diagram:
        fraction_of_positives vs mean_predicted_value across quantile or uniform bins.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

        # Bin histogram counts
        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0.0, 1.0))

        # Expected Calibration Error (ECE) and Max Calibration Error (MCE)
        errors = np.abs(prob_true - prob_pred)
        mce = float(np.max(errors)) if len(errors) > 0 else 0.0

        # Weighted ECE
        weights = bin_counts[:len(prob_true)] / len(y_prob) if len(y_prob) > 0 else np.zeros_like(prob_true)
        ece = float(np.sum(weights * errors)) if len(errors) > 0 else 0.0

        return {
            "fraction_of_positives": prob_true.tolist(),
            "mean_predicted_value": prob_pred.tolist(),
            "bin_counts": bin_counts.tolist(),
            "expected_calibration_error": round(ece, 4),
            "max_calibration_error": round(mce, 4),
        }

    def calibrate_classifier(
        self,
        base_estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "sigmoid",
        cv: int = 3,
    ) -> CalibratedClassifierCV:
        """
        Fit a CalibratedClassifierCV wrapper on top of an estimator.
        Method: 'sigmoid' (Platt scaling) or 'isotonic'.
        """
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv=cv,
        )
        calibrated.fit(X_train, y_train)
        logger.info(f"Fitted CalibratedClassifierCV using method='{method}' across {cv}-fold CV.")
        return calibrated


calibration_service = CalibrationService()
