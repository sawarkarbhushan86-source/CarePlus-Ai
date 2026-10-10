"""
CARE-PULSE AI — Data Validation Service
Comprehensive clinical encounter data quality checks, schema audits, missingness analysis,
duplicate detection, and clinical cohort exclusion rules.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from app.config import settings
from app.schemas.dataset import DataValidationReport, ColumnQualitySummary

logger = logging.getLogger(__name__)

# Essential columns expected in UCI 130-US Hospitals dataset
ESSENTIAL_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "insulin",
    "change",
    "diabetesMed",
    "readmitted",
]

# Discharge disposition codes indicating patient expired or entered hospice
# 11: Expired
# 19: Expired at home (Medicaid/Hospice)
# 20: Expired in a medical facility (Medicaid/Hospice)
# 21: Expired in a place unknown
EXCLUDED_DISCHARGE_IDS = [11, 19, 20, 21]


class DataValidationService:
    """Validates raw and processed healthcare encounter datasets against clinical ML criteria."""

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify presence of essential clinical features."""
        missing_cols = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
        if missing_cols:
            return False, [f"Missing required clinical column: {c}" for c in missing_cols]
        return True, ["All essential clinical features present."]

    def analyze_missingness(self, df: pd.DataFrame, threshold: float = 0.40) -> Dict[str, Any]:
        """Compute missingness rates accounting for '?', 'None', and np.nan."""
        # Replace string missing markers
        cleaned = df.replace(["?", "None", "Unknown/Invalid", ""], np.nan)
        missing_rates = cleaned.isna().mean().to_dict()
        high_missing = [col for col, rate in missing_rates.items() if rate > threshold]

        return {
            "missing_rates": {k: round(float(v), 4) for k, v in missing_rates.items()},
            "high_missingness_columns": high_missing,
        }

    def detect_duplicates_and_repeats(self, df: pd.DataFrame) -> Dict[str, int]:
        """Audit duplicate encounters and patient revisit frequencies."""
        duplicate_encounters = int(df.duplicated(subset=[settings.ENCOUNTER_ID_COLUMN]).sum()) if settings.ENCOUNTER_ID_COLUMN in df.columns else 0
        unique_patients = int(df[settings.PATIENT_ID_COLUMN].nunique()) if settings.PATIENT_ID_COLUMN in df.columns else len(df)
        total_rows = len(df)
        repeat_encounters = total_rows - unique_patients

        return {
            "duplicate_encounters": duplicate_encounters,
            "unique_patients": unique_patients,
            "repeated_encounters": repeat_encounters,
        }

    def evaluate_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect distribution of original readmitted status and 30-day binary target."""
        if settings.TARGET_COLUMN_RAW in df.columns:
            raw_counts = df[settings.TARGET_COLUMN_RAW].value_counts().to_dict()
            total = len(df)
            thirty_day_positive = raw_counts.get("<30", 0)
            rate_30d = round(thirty_day_positive / total if total > 0 else 0.0, 4)
            return {
                "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
                "rate_30d": rate_30d,
            }
        elif settings.TARGET_COLUMN_BINARY in df.columns:
            counts = df[settings.TARGET_COLUMN_BINARY].value_counts().to_dict()
            total = len(df)
            positives = counts.get(1, 0)
            rate = round(positives / total if total > 0 else 0.0, 4)
            return {
                "raw_counts": {str(k): int(v) for k, v in counts.items()},
                "rate_30d": rate,
            }
        return {"raw_counts": {}, "rate_30d": 0.0}

    def identify_clinical_exclusions(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify encounters that must be excluded from 30-day readmission prediction:
        1. Deceased or hospice discharges (readmission is not possible).
        2. Gender 'Unknown/Invalid' (extremely rare records, e.g. 3 rows).
        """
        mask = pd.Series(False, index=df.index)

        if "discharge_disposition_id" in df.columns:
            # Convert to numeric safely
            disp_ids = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
            mask = mask | disp_ids.isin(EXCLUDED_DISCHARGE_IDS)

        if "gender" in df.columns:
            mask = mask | (df["gender"] == "Unknown/Invalid")

        return mask

    def run_full_validation(self, df: pd.DataFrame) -> DataValidationReport:
        """Perform full validation audit returning structured DataValidationReport."""
        schema_valid, schema_msgs = self.validate_schema(df)
        missing_data = self.analyze_missingness(df)
        repeat_data = self.detect_duplicates_and_repeats(df)
        target_data = self.evaluate_target_distribution(df)
        exclusion_mask = self.identify_clinical_exclusions(df)
        excluded_count = int(exclusion_mask.sum())

        messages = list(schema_msgs)
        if repeat_data["repeated_encounters"] > 0:
            messages.append(
                f"Identified {repeat_data['repeated_encounters']} repeated patient visits "
                f"across {repeat_data['unique_patients']} unique patients. "
                "Patient-grouped splitting is strictly required to prevent data leakage."
            )

        if excluded_count > 0:
            messages.append(
                f"Identified {excluded_count} encounters meeting clinical exclusion criteria "
                "(deceased/hospice discharges). These must be excluded from training/evaluation cohorts."
            )

        return DataValidationReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            is_valid=schema_valid,
            missingness_summary=missing_data["missing_rates"],
            high_missingness_columns=missing_data["high_missingness_columns"],
            duplicate_rows_count=repeat_data["duplicate_encounters"],
            unique_patients_count=repeat_data["unique_patients"],
            repeated_patient_encounters_count=repeat_data["repeated_encounters"],
            target_distribution=target_data["raw_counts"],
            target_rate_30d=target_data["rate_30d"],
            excluded_encounters_count=excluded_count,
            validation_messages=messages,
        )

    def get_column_quality_summaries(self, df: pd.DataFrame) -> List[ColumnQualitySummary]:
        """Generate detailed column-by-column quality metrics."""
        cleaned = df.replace(["?", "None", "Unknown/Invalid", ""], np.nan)
        summaries = []
        for col in df.columns:
            non_null = int(cleaned[col].notna().sum())
            null_cnt = int(cleaned[col].isna().sum())
            missing_pct = round((null_cnt / len(df)) * 100, 2)
            unique_cnt = int(df[col].nunique(dropna=False))
            sample_vals = df[col].dropna().unique()[:5].tolist()

            summaries.append(
                ColumnQualitySummary(
                    name=col,
                    dtype=str(df[col].dtype),
                    non_null_count=non_null,
                    null_count=null_cnt,
                    missing_percentage=missing_pct,
                    unique_values=unique_cnt,
                    sample_values=[str(v) for v in sample_vals],
                )
            )
        return summaries


data_validation_service = DataValidationService()
