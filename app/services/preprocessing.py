"""
CARE-PULSE AI — Preprocessing and Feature Engineering Service
Implements clinically grounded ICD-9 diagnosis clustering, medication change aggregations,
prior utilization scoring, patient-grouped train/test splitting, and Scikit-learn pipelines.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import settings

logger = logging.getLogger(__name__)

# Core list of diabetes medications tracked in the encounter record
MEDICATION_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "insulin",
]

EXCLUDED_DISCHARGE_IDS = [11, 19, 20, 21]


def map_icd9_to_category(code: Any) -> str:
    """
    Map ICD-9 codes to high-level clinical disease categories.
    Categories follow healthcare research conventions (Strack et al., 2014):
    - Circulatory: 390–459, 785
    - Respiratory: 460–519, 786
    - Digestive: 520–579, 787
    - Diabetes: 250.xx
    - Injury: 800–999
    - Musculoskeletal: 710–739
    - Genitourinary: 580–629, 788
    - Neoplasms: 140–239
    - Other
    """
    if pd.isna(code) or code in ["?", "None", "", "Unknown"]:
        return "Missing/Unknown"

    code_str = str(code).strip()

    # Diabetes (250 or 250.xx)
    if code_str.startswith("250"):
        return "Diabetes"

    # Handle E and V codes (External causes / Supplemental factors)
    if code_str.startswith("E") or code_str.startswith("V"):
        return "Other"

    # Extract numeric part
    try:
        num = float(code_str.split(".")[0])
    except ValueError:
        return "Other"

    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    elif (460 <= num <= 519) or num == 786:
        return "Respiratory"
    elif (520 <= num <= 579) or num == 787:
        return "Digestive"
    elif (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 800 <= num <= 999:
        return "Injury"
    else:
        return "Other"


class PreprocessingService:
    """Handles healthcare feature engineering, patient grouping, and Scikit-learn pipelines."""

    def filter_clinical_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude deceased/hospice encounters and invalid records."""
        cleaned = df.copy()
        if "discharge_disposition_id" in cleaned.columns:
            disp_num = pd.to_numeric(cleaned["discharge_disposition_id"], errors="coerce")
            cleaned = cleaned[~disp_num.isin(EXCLUDED_DISCHARGE_IDS)]

        if "gender" in cleaned.columns:
            cleaned = cleaned[cleaned["gender"] != "Unknown/Invalid"]

        return cleaned

    def create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 30-day binary readmission target:
        1 if readmitted == '<30', else 0.
        Preserves original 'readmitted' column.
        """
        df = df.copy()
        if settings.TARGET_COLUMN_RAW in df.columns:
            df[settings.TARGET_COLUMN_BINARY] = (df[settings.TARGET_COLUMN_RAW] == settings.POSITIVE_CLASS_VALUE).astype(int)
        elif settings.TARGET_COLUMN_BINARY not in df.columns:
            df[settings.TARGET_COLUMN_BINARY] = 0
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive clinically meaningful variables from raw encounter features."""
        df = df.copy()

        # Clean missing markers
        df = df.replace(["?", "None", ""], np.nan)

        # 1. Map ICD-9 diagnoses
        for diag_col in ["diag_1", "diag_2", "diag_3"]:
            if diag_col in df.columns:
                df[f"{diag_col}_group"] = df[diag_col].apply(map_icd9_to_category)

        # 2. Prior utilization metrics
        for util_col in ["number_outpatient", "number_emergency", "number_inpatient"]:
            if util_col in df.columns:
                df[util_col] = pd.to_numeric(df[util_col], errors="coerce").fillna(0)

        df["total_prior_visits"] = (
            df.get("number_outpatient", 0) + df.get("number_emergency", 0) + df.get("number_inpatient", 0)
        )
        df["has_prior_inpatient"] = (df.get("number_inpatient", 0) > 0).astype(int)
        df["has_prior_emergency"] = (df.get("number_emergency", 0) > 0).astype(int)

        # 3. Glycemic and Lab test indicators
        if "A1Cresult" in df.columns:
            df["a1c_tested"] = df["A1Cresult"].notna().astype(int)
            df["a1c_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
        else:
            df["a1c_tested"] = 0
            df["a1c_abnormal"] = 0

        if "max_glu_serum" in df.columns:
            df["glucose_tested"] = df["max_glu_serum"].notna().astype(int)
            df["glucose_abnormal"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)
        else:
            df["glucose_tested"] = 0
            df["glucose_abnormal"] = 0

        # 4. Medication regimen metrics
        if "change" in df.columns:
            df["med_changed"] = (df["change"] == "Ch").astype(int)
        else:
            df["med_changed"] = 0

        if "diabetesMed" in df.columns:
            df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)
        else:
            df["on_diabetes_med"] = 0

        # Count active medications
        active_meds = pd.Series(0, index=df.index)
        for med in MEDICATION_COLS:
            if med in df.columns:
                active_meds += (df[med].isin(["Steady", "Up", "Down"])).astype(int)
        df["num_active_diabetes_meds"] = active_meds

        # Insulin indicator
        if "insulin" in df.columns:
            df["on_insulin"] = (df["insulin"].isin(["Steady", "Up", "Down"])).astype(int)
        else:
            df["on_insulin"] = 0

        # Clean numerical encounter columns
        for num_col in [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
        ]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

        return df

    def get_feature_lists(self) -> Tuple[List[str], List[str]]:
        """Return designated numerical and categorical feature names for modeling."""
        num_cols = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "total_prior_visits",
            "has_prior_inpatient",
            "has_prior_emergency",
            "a1c_tested",
            "a1c_abnormal",
            "glucose_tested",
            "glucose_abnormal",
            "med_changed",
            "on_diabetes_med",
            "num_active_diabetes_meds",
            "on_insulin",
        ]

        cat_cols = [
            "race",
            "gender",
            "age",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "diag_1_group",
            "diag_2_group",
            "diag_3_group",
            "max_glu_serum",
            "A1Cresult",
            "insulin",
        ]

        return num_cols, cat_cols

    def build_preprocessor_pipeline(self, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        """Construct Scikit-learn ColumnTransformer with median/mode imputation and scaling."""
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            remainder="drop",
        )
        return preprocessor

    def prepare_modeling_data(
        self, df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Execute full preparation workflow:
        1. Clinical exclusions
        2. Binary target generation
        3. Feature engineering
        4. Patient-grouped train/test split
        """
        df_filtered = self.filter_clinical_exclusions(df)
        df_targeted = self.create_target_column(df_filtered)
        df_features = self.engineer_features(df_targeted)

        num_cols, cat_cols = self.get_feature_lists()

        # Ensure all columns exist
        available_num_cols = [c for c in num_cols if c in df_features.columns]
        available_cat_cols = [c for c in cat_cols if c in df_features.columns]

        # Convert categorical ID columns to string for OneHotEncoder
        for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(str)

        all_feature_cols = available_num_cols + available_cat_cols
        X = df_features[all_feature_cols].copy()
        y = df_features[settings.TARGET_COLUMN_BINARY].copy()

        # Patient-grouped splitting
        if settings.PATIENT_ID_COLUMN in df_features.columns:
            groups = df_features[settings.PATIENT_ID_COLUMN].values
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
        else:
            # Fallback to standard random index split
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=y
            )

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        logger.info(
            f"Prepared modeling dataset. Train: {len(X_train)} encounters, "
            f"Test: {len(X_test)} encounters. Positive prevalence in test: {y_test.mean():.4f}"
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "num_cols": available_num_cols,
            "cat_cols": available_cat_cols,
            "feature_columns": all_feature_cols,
            "raw_train_df": df_features.iloc[train_idx].reset_index(drop=True),
            "raw_test_df": df_features.iloc[test_idx].reset_index(drop=True),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }


preprocessing_service = PreprocessingService()
