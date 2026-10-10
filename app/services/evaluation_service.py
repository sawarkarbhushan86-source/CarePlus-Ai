"""
CARE-PULSE AI — Model Evaluation Service
Comprehensive clinical classification and probabilistic evaluation metrics:
ROC-AUC, PR-AUC, Brier Score, Brier Skill Score, Expected Calibration Error (ECE),
Sensitivity/Recall, Specificity, Positive Predictive Value, and Confusion Matrices.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from app.schemas.experiment import ModelMetrics

logger = logging.getLogger(__name__)


class EvaluationService:
    """Evaluates classification discrimination and probabilistic reliability."""

    def compute_expected_calibration_error(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE):
        ECE = sum_b ( |acc(b) - conf(b)| * (|b| / N) )
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1

        ece = 0.0
        n_samples = len(y_true)

        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                bin_weight = np.sum(mask) / n_samples
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        return float(ece)

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.15,
    ) -> ModelMetrics:
        """Compute full suite of clinical discrimination and calibration metrics."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)

        # Discrimination metrics
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Probabilistic Calibration Metrics
        brier = float(brier_score_loss(y_true, y_prob))
        base_rate = float(np.mean(y_true))
        brier_ref = float(brier_score_loss(y_true, np.full_like(y_true, base_rate, dtype=float)))
        brier_skill = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0

        ece = self.compute_expected_calibration_error(y_true, y_prob, n_bins=10)

        # Optimal Youden's threshold
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr
        opt_thresh = float(roc_thresh[np.argmax(youden_index)])

        return ModelMetrics(
            model_name=model_name,
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            specificity=round(specificity, 4),
            brier_score=round(brier, 4),
            brier_skill_score=round(brier_skill, 4),
            expected_calibration_error=round(ece, 4),
            confusion_matrix={
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            optimal_threshold=round(opt_thresh, 4),
        )

    def get_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Generate ROC and PR curve data points for interactive Plotly charts."""
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

        # Downsample for lightweight payload if large
        step = max(1, len(fpr) // 100)
        return {
            "roc": {
                "fpr": fpr[::step].tolist(),
                "tpr": tpr[::step].tolist(),
                "thresholds": roc_thresh[::step].tolist(),
            },
            "pr": {
                "precision": prec[::step].tolist(),
                "recall": rec[::step].tolist(),
                "thresholds": pr_thresh[::step].tolist() if len(pr_thresh) > 0 else [],
            },
        }

    def compute_bootstrap_confidence_intervals(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 500, random_state: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """Compute non-parametric 95% bootstrap confidence intervals for primary performance metrics."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        rng = np.random.RandomState(random_state)
        n = len(y_true)

        boot_aucs = []
        boot_pr_aucs = []
        boot_briers = []
        boot_eces = []

        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, n)
            if len(np.unique(y_true[indices])) < 2:
                continue
            y_b = y_true[indices]
            p_b = y_prob[indices]
            boot_aucs.append(roc_auc_score(y_b, p_b))
            boot_pr_aucs.append(average_precision_score(y_b, p_b))
            boot_briers.append(brier_score_loss(y_b, p_b))
            boot_eces.append(self.compute_expected_calibration_error(y_b, p_b, n_bins=10))

        def ci_dict(arr, point_est):
            low = float(np.percentile(arr, 2.5)) if arr else point_est
            high = float(np.percentile(arr, 97.5)) if arr else point_est
            return {
                "point_estimate": round(float(point_est), 4),
                "ci_95_low": round(low, 4),
                "ci_95_high": round(high, 4),
            }

        point_auc = roc_auc_score(y_true, y_prob)
        point_pr = average_precision_score(y_true, y_prob)
        point_brier = brier_score_loss(y_true, y_prob)
        point_ece = self.compute_expected_calibration_error(y_true, y_prob, n_bins=10)

        return {
            "roc_auc": ci_dict(boot_aucs, point_auc),
            "pr_auc": ci_dict(boot_pr_aucs, point_pr),
            "brier_score": ci_dict(boot_briers, point_brier),
            "ece": ci_dict(boot_eces, point_ece),
        }

    def evaluate_threshold_grid(
        self, y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Compute performance trade-offs across a grid of decision thresholds."""
        if thresholds is None:
            thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        n_pos = int(np.sum(y_true))

        grid = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            flagged = int(tp + fp)
            flagged_pct = (flagged / n) * 100.0 if n > 0 else 0.0
            workload_per_1000 = round((flagged / n) * 1000) if n > 0 else 0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            precision = float(tp / flagged) if flagged > 0 else 0.0
            missed_cases = int(fn)
            missed_pct = float((fn / n_pos) * 100.0) if n_pos > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            grid.append({
                "threshold": round(float(t), 2),
                "flagged_patients": int(flagged),
                "flagged_percentage": round(float(flagged_pct), 1),
                "workload_per_1000_discharges": int(workload_per_1000),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "recall_sensitivity": round(float(recall), 4),
                "specificity": round(float(specificity), 4),
                "precision_ppv": round(float(precision), 4),
                "missed_readmission_count": int(missed_cases),
                "missed_readmissions_percentage": round(float(missed_pct), 1),
                "f1_score": round(float(f1), 4),
            })
        return grid


evaluation_service = EvaluationService()
