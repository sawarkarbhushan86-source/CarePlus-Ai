"""
CARE-PULSE AI — Biostatistical Analysis Service
Implements formal hypothesis testing, Chi-Square tests of independence, Cramér's V,
Mann-Whitney U tests, Cohen's d, and Odds Ratios with 95% Wald confidence intervals.
"""

import math
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from app.config import settings
from app.services.preprocessing import preprocessing_service

logger = logging.getLogger(__name__)


class StatisticsService:
    """Provides epidemiological and statistical inference for healthcare encounter risk factors."""

    def compute_odds_ratio(
        self, df: pd.DataFrame, exposure_col: str, target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute Odds Ratio (OR) with 95% Wald Confidence Interval for a binary exposure.
        OR = (a * d) / (b * c)
        where:
        a = exposed & readmitted
        b = exposed & not readmitted
        c = unexposed & readmitted
        d = unexposed & not readmitted
        """
        target = target_col or settings.TARGET_COLUMN_BINARY
        if exposure_col not in df.columns or target not in df.columns:
            return {"error": f"Columns {exposure_col} or {target} missing from dataset."}

        # Binarize exposure if boolean or 0/1
        exp_binary = df[exposure_col].astype(bool).astype(int)
        tgt_binary = df[target].astype(bool).astype(int)

        table = pd.crosstab(exp_binary, tgt_binary)

        # Ensure 2x2 table format
        a = int(table.loc[1, 1]) if (1 in table.index and 1 in table.columns) else 0
        b = int(table.loc[1, 0]) if (1 in table.index and 0 in table.columns) else 0
        c = int(table.loc[0, 1]) if (0 in table.index and 1 in table.columns) else 0
        d = int(table.loc[0, 0]) if (0 in table.index and 0 in table.columns) else 0

        # Haldane-Anscombe correction if any cell is 0
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5

        odds_ratio = (a * d) / (b * c)
        se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = 1.96  # 95% CI

        ci_low = math.exp(math.log(odds_ratio) - z * se_log_or)
        ci_high = math.exp(math.log(odds_ratio) + z * se_log_or)

        # P-value via Fisher's exact test or chi-square
        res_fisher = stats.fisher_exact([[int(a), int(b)], [int(c), int(d)]])
        p_val = float(res_fisher.pvalue)

        return {
            "exposure": exposure_col,
            "odds_ratio": round(float(odds_ratio), 3),
            "ci_95_low": round(float(ci_low), 3),
            "ci_95_high": round(float(ci_high), 3),
            "p_value": p_val,
            "significant_at_05": bool(p_val < 0.05),
            "table_counts": {"exposed_cases": int(a), "exposed_controls": int(b), "unexposed_cases": int(c), "unexposed_controls": int(d)},
        }

    def compute_all_clinical_odds_ratios(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate odds ratios for standard clinical readmission risk exposures."""
        df = preprocessing_service.create_target_column(df)
        df = preprocessing_service.engineer_features(df)

        # Define exposures
        df_exp = df.copy()
        df_exp["prior_inpatient_visit"] = (df_exp["number_inpatient"] > 0).astype(int)
        df_exp["prior_emergency_visit"] = (df_exp["number_emergency"] > 0).astype(int)
        df_exp["length_of_stay_over_7d"] = (df_exp["time_in_hospital"] > 7).astype(int)
        df_exp["polypharmacy_over_15meds"] = (df_exp["num_medications"] > 15).astype(int)
        df_exp["high_diagnoses_over_7"] = (df_exp["number_diagnoses"] > 7).astype(int)
        df_exp["insulin_administered"] = (df_exp["on_insulin"] == 1).astype(int)
        df_exp["medication_changed"] = (df_exp["med_changed"] == 1).astype(int)
        df_exp["a1c_untested"] = (df_exp["a1c_tested"] == 0).astype(int)

        exposures = [
            ("prior_inpatient_visit", "Prior Inpatient Admission (>=1)"),
            ("prior_emergency_visit", "Prior Emergency Visit (>=1)"),
            ("length_of_stay_over_7d", "Prolonged Stay (>7 Days)"),
            ("polypharmacy_over_15meds", "Polypharmacy (>15 Medications)"),
            ("high_diagnoses_over_7", "Multimorbidity (>7 Diagnoses)"),
            ("insulin_administered", "Insulin Prescribed/Administered"),
            ("medication_changed", "Diabetic Medication Adjusted"),
            ("a1c_untested", "HbA1c Lab Untested"),
        ]

        results = []
        for col_name, label in exposures:
            or_dict = self.compute_odds_ratio(df_exp, col_name)
            or_dict["label"] = label
            results.append(or_dict)

        return results

    def compute_chi_square_test(
        self, df: pd.DataFrame, categorical_col: str, target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Pearson Chi-Square test of independence (or Fisher's exact test if assumptions violated)
        and compute Cramér's V effect size.
        Validates Cochran's assumptions: expected frequencies >= 1 in all cells, and < 5 in at most 20% of cells.
        """
        from app.services.preprocessing import map_icd9_to_category

        df = df.copy()
        target = target_col or settings.TARGET_COLUMN_BINARY

        # Ensure binary target exists
        if target not in df.columns:
            df = preprocessing_service.create_target_column(df)

        # Automatically resolve engineered clinical features if raw inputs are present
        if categorical_col not in df.columns:
            if categorical_col == "diag_1_group" and "diag_1" in df.columns:
                df["diag_1_group"] = df["diag_1"].apply(map_icd9_to_category)
            elif categorical_col in ["on_insulin", "med_changed", "a1c_tested", "a1c_abnormal"]:
                df = preprocessing_service.engineer_clinical_features(df)

        # Input validation
        if categorical_col not in df.columns or target not in df.columns:
            return {
                "success": False,
                "factor": categorical_col,
                "error": f"Clinical variable '{categorical_col}' not found in dataset.",
                "chi2_statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "cramers_v": 0.0,
                "effect_size_magnitude": "Invalid",
                "test_used": "None",
                "assumptions_met": False,
                "significant_at_05": False,
            }

        # Filter complete cases
        valid_df = df[[categorical_col, target]].dropna()
        if len(valid_df) == 0:
            return {
                "success": False,
                "factor": categorical_col,
                "error": f"No non-null records available for '{categorical_col}'.",
                "chi2_statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "cramers_v": 0.0,
                "effect_size_magnitude": "Invalid",
                "test_used": "None",
                "assumptions_met": False,
                "significant_at_05": False,
            }

        contingency = pd.crosstab(valid_df[categorical_col], valid_df[target])

        # Dimension validation: must be at least 2x2
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {
                "success": False,
                "factor": categorical_col,
                "error": f"Variable '{categorical_col}' has fewer than 2 categories ({contingency.shape[0]} found). A 2x2 or larger table is required.",
                "chi2_statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": 0,
                "cramers_v": 0.0,
                "effect_size_magnitude": "Invalid",
                "test_used": "None",
                "assumptions_met": False,
                "significant_at_05": False,
            }

        # Standard Pearson Chi-Square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        # Cochran's rule audit
        n_cells = expected.size
        cells_under_5 = int(np.sum(expected < 5.0))
        pct_under_5 = float((cells_under_5 / n_cells) * 100.0) if n_cells > 0 else 0.0
        has_cells_under_1 = bool(np.any(expected < 1.0))
        cochran_met = (not has_cells_under_1) and (pct_under_5 <= 20.0)

        test_used = "Pearson Chi-Square Test of Independence"

        # Alternative test if assumptions violated on 2x2
        if not cochran_met and contingency.shape == (2, 2):
            _, p_fisher = stats.fisher_exact(contingency.values)
            p = float(p_fisher)
            test_used = "Fisher's Exact Test (Alternative for 2x2 with expected counts < 5)"

        # Cramér's V
        n = float(contingency.values.sum())
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if (n > 0 and min_dim > 0) else 0.0

        return {
            "success": True,
            "factor": categorical_col,
            "chi2_statistic": round(float(chi2), 3),
            "p_value": float(p),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(float(cramers_v), 4),
            "effect_size_magnitude": self._cramers_v_interpretation(cramers_v),
            "significant_at_05": bool(p < 0.05),
            "test_used": test_used,
            "assumptions_met": cochran_met,
            "cochran_warning": (
                None
                if cochran_met
                else f"Cochran rule violated: {pct_under_5:.1f}% of cells have expected frequency < 5 (max allowed: 20%)."
            ),
            "total_observations": int(n),
            "categories_evaluated": list(contingency.index),
        }

    def compute_continuous_comparison(
        self, df: pd.DataFrame, numeric_col: str, target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test and Cohen's d between readmitted and non-readmitted cohorts."""
        df = preprocessing_service.create_target_column(df)
        target = target_col or settings.TARGET_COLUMN_BINARY
        if numeric_col not in df.columns or target not in df.columns:
            return {"error": f"Columns {numeric_col} or {target} missing."}

        group_0 = pd.to_numeric(df[df[target] == 0][numeric_col], errors="coerce").dropna()
        group_1 = pd.to_numeric(df[df[target] == 1][numeric_col], errors="coerce").dropna()

        # Mann-Whitney U
        u_stat, p_val = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")

        # Cohen's d
        n1, n2 = len(group_1), len(group_0)
        s1, s2 = group_1.std(), group_0.std()
        s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
        cohens_d = (group_1.mean() - group_0.mean()) / s_pooled if s_pooled > 0 else 0.0

        return {
            "metric": numeric_col,
            "readmitted_mean": round(float(group_1.mean()), 2),
            "non_readmitted_mean": round(float(group_0.mean()), 2),
            "readmitted_median": round(float(group_1.median()), 2),
            "non_readmitted_median": round(float(group_0.median()), 2),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "cohens_d": round(float(cohens_d), 3),
            "significant_at_05": bool(p_val < 0.05),
        }

    def _cramers_v_interpretation(self, v: float) -> str:
        if v < 0.05:
            return "Negligible"
        elif v < 0.15:
            return "Small"
        elif v < 0.25:
            return "Moderate"
        else:
            return "Strong"


statistics_service = StatisticsService()
