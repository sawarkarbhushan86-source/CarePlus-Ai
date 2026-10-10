"""
Unit tests for CARE-PULSE AI Training Service and Preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from app.services.preprocessing import preprocessing_service, map_icd9_to_category
from app.services.training_service import training_service
from app.config import settings


def test_icd9_mapping():
    assert map_icd9_to_category("250.02") == "Diabetes"
    assert map_icd9_to_category("250") == "Diabetes"
    assert map_icd9_to_category("428") == "Circulatory"
    assert map_icd9_to_category("486") == "Respiratory"
    assert map_icd9_to_category("584") == "Genitourinary"
    assert map_icd9_to_category("820") == "Injury"
    assert map_icd9_to_category("?") == "Missing/Unknown"
    assert map_icd9_to_category(None) == "Missing/Unknown"


def test_patient_grouped_splitting():
    # Construct synthetic multi-encounter dataset with repeated patients
    np.random.seed(42)
    n_encounters = 200
    patient_ids = np.random.choice(np.arange(100, 150), size=n_encounters)

    df_synth = pd.DataFrame(
        {
            "encounter_id": np.arange(1000, 1000 + n_encounters),
            "patient_nbr": patient_ids,
            "race": ["Caucasian"] * n_encounters,
            "gender": ["Female"] * n_encounters,
            "age": ["[60-70)"] * n_encounters,
            "admission_type_id": [1] * n_encounters,
            "discharge_disposition_id": [1] * n_encounters,
            "admission_source_id": [7] * n_encounters,
            "time_in_hospital": np.random.randint(1, 10, size=n_encounters),
            "num_lab_procedures": np.random.randint(10, 80, size=n_encounters),
            "num_procedures": [1] * n_encounters,
            "num_medications": np.random.randint(5, 25, size=n_encounters),
            "number_outpatient": [0] * n_encounters,
            "number_emergency": [0] * n_encounters,
            "number_inpatient": np.random.choice([0, 1, 2], size=n_encounters),
            "diag_1": ["250"] * n_encounters,
            "diag_2": ["401"] * n_encounters,
            "diag_3": ["272"] * n_encounters,
            "number_diagnoses": [6] * n_encounters,
            "max_glu_serum": ["None"] * n_encounters,
            "A1Cresult": [">8"] * n_encounters,
            "metformin": ["No"] * n_encounters,
            "insulin": ["Up"] * n_encounters,
            "change": ["Ch"] * n_encounters,
            "diabetesMed": ["Yes"] * n_encounters,
            "readmitted": np.random.choice(["NO", ">30", "<30"], size=n_encounters, p=[0.5, 0.35, 0.15]),
        }
    )

    split_res = preprocessing_service.prepare_modeling_data(df_synth, test_size=0.20, random_state=42)

    train_patients = set(split_res["raw_train_df"]["patient_nbr"])
    test_patients = set(split_res["raw_test_df"]["patient_nbr"])

    # Patient grouped guarantee: zero intersection between train and test patients
    intersection = train_patients.intersection(test_patients)
    assert len(intersection) == 0, f"Found leaked patients across train/test: {intersection}"


def test_model_loading_and_prediction():
    # Ensure Calibrated XGBoost or Logistic Regression can be loaded and outputs probability
    model = training_service.load_model("Logistic Regression")
    assert model is not None
    assert hasattr(model, "predict_proba")
