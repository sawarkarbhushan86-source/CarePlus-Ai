"""
Unit tests for CARE-PULSE AI Data Validation Service.
"""

import pytest
import pandas as pd
import numpy as np

from app.services.data_validation import data_validation_service, ESSENTIAL_COLUMNS
from app.services.data_loader import data_loader_service


@pytest.fixture
def sample_encounter_df():
    """Create a minimal valid encounter dataframe for unit testing."""
    data = {
        "encounter_id": [101, 102, 103, 104, 105],
        "patient_nbr": [1001, 1002, 1001, 1003, 1004],  # 1001 has repeat encounter
        "race": ["Caucasian", "AfricanAmerican", "Caucasian", "Hispanic", "?"],
        "gender": ["Female", "Male", "Female", "Male", "Unknown/Invalid"],
        "age": ["[50-60)", "[60-70)", "[50-60)", "[70-80)", "[80-90)"],
        "admission_type_id": [1, 2, 1, 3, 1],
        "discharge_disposition_id": [1, 3, 6, 11, 1],  # 11 = expired (clinical exclusion)
        "admission_source_id": [7, 1, 7, 2, 7],
        "time_in_hospital": [3, 5, 2, 8, 4],
        "num_lab_procedures": [40, 52, 33, 60, 45],
        "num_procedures": [1, 2, 0, 1, 1],
        "num_medications": [12, 18, 14, 22, 16],
        "number_outpatient": [0, 1, 0, 0, 2],
        "number_emergency": [0, 0, 1, 0, 0],
        "number_inpatient": [0, 1, 1, 2, 0],
        "diag_1": ["250.02", "428", "486", "250", "401"],
        "diag_2": ["401", "250", "428", "414", "272"],
        "diag_3": ["272", "401", "250", "428", "250"],
        "number_diagnoses": [5, 7, 6, 9, 7],
        "max_glu_serum": ["None", "Norm", ">200", "None", "None"],
        "A1Cresult": [">8", "Norm", "None", ">7", "None"],
        "metformin": ["No", "Steady", "No", "No", "Steady"],
        "insulin": ["Up", "No", "Steady", "Down", "No"],
        "change": ["Ch", "No", "No", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "Yes", "Yes", "Yes"],
        "readmitted": ["<30", "NO", ">30", "<30", "NO"],
    }
    return pd.DataFrame(data)


def test_schema_validation_success(sample_encounter_df):
    valid, msgs = data_validation_service.validate_schema(sample_encounter_df)
    assert valid is True
    assert "All essential clinical features present." in msgs[0]


def test_schema_validation_failure(sample_encounter_df):
    corrupted_df = sample_encounter_df.drop(columns=["discharge_disposition_id", "A1Cresult"])
    valid, msgs = data_validation_service.validate_schema(corrupted_df)
    assert valid is False
    assert len(msgs) == 2


def test_duplicate_and_repeat_detection(sample_encounter_df):
    repeat_stats = data_validation_service.detect_duplicates_and_repeats(sample_encounter_df)
    assert repeat_stats["duplicate_encounters"] == 0
    assert repeat_stats["unique_patients"] == 4
    assert repeat_stats["repeated_encounters"] == 1  # 5 rows - 4 unique patients = 1 repeat


def test_clinical_exclusion_identification(sample_encounter_df):
    exclusions = data_validation_service.identify_clinical_exclusions(sample_encounter_df)
    # Row 3 has discharge_disposition_id = 11 (Expired)
    # Row 4 has gender = 'Unknown/Invalid'
    assert exclusions.iloc[3] is True or exclusions.iloc[3] == 1
    assert exclusions.iloc[4] is True or exclusions.iloc[4] == 1
    assert int(exclusions.sum()) == 2


def test_missingness_analysis(sample_encounter_df):
    miss_res = data_validation_service.analyze_missingness(sample_encounter_df)
    assert "missing_rates" in miss_res
    assert isinstance(miss_res["missing_rates"], dict)


def test_full_validation_report(sample_encounter_df):
    report = data_validation_service.run_full_validation(sample_encounter_df)
    assert report.total_rows == 5
    assert report.total_columns == 26
    assert report.is_valid is True
    assert report.unique_patients_count == 4
    assert report.excluded_encounters_count == 2
    assert report.target_rate_30d == 0.40  # 2 of 5 are <30
