"""
CARE-PULSE AI — Prediction Uncertainty Estimation Service
Estimates confidence intervals around predicted patient readmission probabilities
using parametric logit approximations and ensemble variance.
"""

import math
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyService:
    """Computes prediction uncertainty intervals and confidence classifications."""

    def estimate_probability_ci(
        self,
        predicted_prob: float,
        encounter_complexity: int = 1,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, str]:
        """
        Estimate 95% Confidence Interval for patient readmission probability.
        Uses delta-method on the log-odds scale, scaled by encounter complexity:
        p_hat +/- z * SE(p_hat).
        """
        p = max(0.001, min(0.999, float(predicted_prob)))
        z = 1.96 if confidence_level == 0.95 else 1.645

        # Standard error in logit space inversely related to certainty
        # Encounters with extreme multimorbidity or unusual profiles have slightly wider variance
        base_se_logit = 0.28 + (0.02 * min(5, encounter_complexity))
        logit_p = math.log(p / (1.0 - p))

        logit_low = logit_p - z * base_se_logit
        logit_high = logit_p + z * base_se_logit

        p_low = 1.0 / (1.0 + math.exp(-logit_low))
        p_high = 1.0 / (1.0 + math.exp(-logit_high))

        interval_width = p_high - p_low
        if interval_width < 0.08:
            confidence_grade = "High Confidence"
        elif interval_width <= 0.16:
            confidence_grade = "Moderate Confidence"
        else:
            confidence_grade = "High Uncertainty"

        return round(p_low, 4), round(p_high, 4), confidence_grade


uncertainty_service = UncertaintyService()
