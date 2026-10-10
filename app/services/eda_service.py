"""
CARE-PULSE AI — Exploratory Data Analysis (EDA) Service
Calculates healthcare encounter descriptive statistics, readmission prevalence breakdowns,
and demographic / clinical utilization distributions.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from app.config import settings
from app.services.preprocessing import preprocessing_service

logger = logging.getLogger(__name__)


class EDAService:
    """Computes clinical summary statistics and cohort distributions for the UI and API."""

    def compute_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute key healthcare indicators across the encounter population."""
        # Ensure target and engineered features are present
        df = preprocessing_service.create_target_column(df)
        df = preprocessing_service.engineer_features(df)

        total_encounters = len(df)
        target_col = settings.TARGET_COLUMN_BINARY
        prevalence_30d = float(df[target_col].mean()) if target_col in df.columns else 0.0

        los_mean = float(df["time_in_hospital"].mean()) if "time_in_hospital" in df.columns else 0.0
        los_median = float(df["time_in_hospital"].median()) if "time_in_hospital" in df.columns else 0.0

        meds_mean = float(df["num_medications"].mean()) if "num_medications" in df.columns else 0.0
        labs_mean = float(df["num_lab_procedures"].mean()) if "num_lab_procedures" in df.columns else 0.0
        procs_mean = float(df["num_procedures"].mean()) if "num_procedures" in df.columns else 0.0

        # Subgroup readmission rates
        readm_by_age = self._group_rate(df, "age", target_col)
        readm_by_race = self._group_rate(df, "race", target_col)
        readm_by_diag = self._group_rate(df, "diag_1_group", target_col)
        readm_by_insulin = self._group_rate(df, "insulin", target_col)
        readm_by_a1c = self._group_rate(df, "A1Cresult", target_col)

        # Prior inpatient visits impact
        prior_inp = df["number_inpatient"].clip(upper=4).astype(str)
        df_temp = df.copy()
        df_temp["prior_inpatient_binned"] = prior_inp
        readm_by_prior_inp = self._group_rate(df_temp, "prior_inpatient_binned", target_col)

        return {
            "total_encounters": total_encounters,
            "readmission_30d_prevalence": round(prevalence_30d, 4),
            "readmission_30d_percentage": round(prevalence_30d * 100, 2),
            "avg_length_of_stay": round(los_mean, 2),
            "median_length_of_stay": round(los_median, 2),
            "avg_num_medications": round(meds_mean, 2),
            "avg_num_lab_procedures": round(labs_mean, 2),
            "avg_num_procedures": round(procs_mean, 2),
            "readmission_by_age": readm_by_age,
            "readmission_by_race": readm_by_race,
            "readmission_by_diag": readm_by_diag,
            "readmission_by_insulin": readm_by_insulin,
            "readmission_by_a1c": readm_by_a1c,
            "readmission_by_prior_inpatient": readm_by_prior_inp,
        }

    def compute_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Compute frequency distribution of a specified column."""
        if column not in df.columns:
            return {"error": f"Column {column} not found"}

        counts = df[column].value_counts(dropna=False).head(20).to_dict()
        total = len(df)
        proportions = {str(k): round(float(v) / total, 4) for k, v in counts.items()}

        return {
            "column": column,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": proportions,
        }

    def compute_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Pearson correlation for core numeric clinical variables."""
        df = preprocessing_service.create_target_column(df)
        num_cols = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            settings.TARGET_COLUMN_BINARY,
        ]
        available = [c for c in num_cols if c in df.columns]
        corr = df[available].apply(pd.to_numeric, errors="coerce").corr().round(3)

        return {
            "columns": available,
            "matrix": corr.values.tolist(),
        }

    def _group_rate(self, df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, float]:
        """Helper to calculate readmission event rate grouped by categorical attribute."""
        if group_col not in df.columns or target_col not in df.columns:
            return {}
        rates = df.groupby(group_col, observed=True)[target_col].mean().round(4).to_dict()
        return {str(k): float(v) for k, v in rates.items()}


eda_service = EDAService()
