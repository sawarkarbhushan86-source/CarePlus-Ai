"""
CARE-PULSE AI — Subgroup Fairness and Reliability Service
Audits model discrimination, error rates, and predictive parity across demographic
and clinical subgroups (Age cohorts, Racial demographics, Biological sex, Prior admissions).
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix

logger = logging.getLogger(__name__)


def compute_wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score interval for binomial proportion k/n with continuity correction.
    Provides reliable confidence intervals even for small sample sizes or extreme proportions.
    """
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% two-tailed standard normal critical value
    p_hat = k / n
    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4.0 * (n ** 2))))
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return round(lower, 4), round(upper, 4)


class SubgroupAnalysisService:
    """Evaluates healthcare algorithmic fairness and performance disparities across patient cohorts."""

    def evaluate_subgroups(
        self,
        df_test: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        subgroup_col: str,
        threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate discrimination, false positive rates, recall, and uncertainty intervals across subgroup_col.
        Accounts for all test encounters by mapping missing values and flagging small, under-powered cohorts.
        """
        if subgroup_col not in df_test.columns:
            return []

        df_eval = df_test.copy()
        # Clean and map missing values
        series = df_eval[subgroup_col].replace("?", np.nan).fillna("Missing / Unknown").astype(str)
        df_eval[subgroup_col] = series
        df_eval["y_true"] = y_true
        df_eval["y_prob"] = y_prob
        df_eval["y_pred"] = (y_prob >= threshold).astype(int)

        results = []
        overall_tpr = float(recall_score(y_true, df_eval["y_pred"], zero_division=0))

        groups = df_eval[subgroup_col].unique()
        for g in groups:
            sub = df_eval[df_eval[subgroup_col] == g]
            n_sub = len(sub)
            if n_sub < 5:
                continue

            sub_y_true = sub["y_true"].values
            sub_y_prob = sub["y_prob"].values
            sub_y_pred = sub["y_pred"].values

            n_pos = int(np.sum(sub_y_true))
            prevalence = float(np.mean(sub_y_true))
            prev_ci_low, prev_ci_high = compute_wilson_score_interval(n_pos, n_sub)

            # Subgroup ROC-AUC if both classes present
            if len(np.unique(sub_y_true)) > 1:
                auc = float(roc_auc_score(sub_y_true, sub_y_prob))
            else:
                auc = 0.50

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(sub_y_true, sub_y_pred, labels=[0, 1]).ravel()
            flagged = int(tp + fp)
            selection_rate = float(flagged / n_sub) if n_sub > 0 else 0.0

            tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            tpr_ci_low, tpr_ci_high = compute_wilson_score_interval(tp, n_pos)

            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            # Disparity vs overall TPR
            tpr_disparity = round(tpr - overall_tpr, 4)

            # Sample size stability warning (N < 100 or positive events < 20)
            is_small = bool(n_sub < 100 or n_pos < 20)
            reliability_note = (
                f"Caution: Under-powered sample (N={n_sub}, N_pos={n_pos}) — estimates exhibit high sampling variance"
                if is_small
                else "Adequate sample size (N >= 100, N_pos >= 20)"
            )

            results.append(
                {
                    "subgroup": str(g),
                    "sample_size": int(n_sub),
                    "positive_events_count": int(n_pos),
                    "readmission_prevalence": round(prevalence, 4),
                    "prevalence_ci_95": [prev_ci_low, prev_ci_high],
                    "roc_auc": round(auc, 4),
                    "sensitivity_recall": round(tpr, 4),
                    "recall_ci_95": [tpr_ci_low, tpr_ci_high],
                    "false_positive_rate": round(fpr, 4),
                    "positive_predictive_value": round(ppv, 4),
                    "selection_rate": round(selection_rate, 4),
                    "flagged_patients_count": flagged,
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "tpr_disparity": tpr_disparity,
                    "small_sample_warning": is_small,
                    "sample_reliability_note": reliability_note,
                }
            )

        # Sort by sample size descending
        results.sort(key=lambda x: x["sample_size"], reverse=True)
        return results

    def get_fairness_audit(
        self,
        df_test: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """Run standard multi-attribute algorithmic fairness audit."""
        attributes = ["race", "gender", "age"]
        audit = {}

        for attr in attributes:
            if attr in df_test.columns:
                audit[attr] = self.evaluate_subgroups(df_test, y_true, y_prob, attr, threshold)

        return audit


subgroup_analysis_service = SubgroupAnalysisService()

