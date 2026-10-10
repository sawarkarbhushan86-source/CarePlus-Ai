"""
Unit tests for CARE-PULSE AI Prediction Service.
"""

import pytest
from app.schemas.prediction import PatientEncounterInput, PredictionResult, BatchPredictionResult
from app.services.prediction_service import prediction_service


@pytest.fixture
def test_encounter():
    return PatientEncounterInput(
        age="[60-70)",
        race="Caucasian",
        gender="Female",
        time_in_hospital=4,
        number_inpatient=1,
        number_emergency=0,
        num_medications=16,
        number_diagnoses=7,
        A1Cresult=">8",
        max_glu_serum="None",
        insulin="Up",
        diag_1="250.02",
        change="Ch",
        diabetesMed="Yes",
    )


def test_single_encounter_prediction(test_encounter):
    result = prediction_service.predict_encounter(test_encounter, model_name="Calibrated XGBoost", threshold=0.15)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.readmission_probability <= 1.0
    assert result.risk_tier in ["Low Risk", "Moderate Risk", "High Risk"]
    assert 0.0 <= result.confidence_interval_low <= result.confidence_interval_high <= 1.0
    assert result.decision_threshold == 0.15
    assert len(result.recommended_interventions) > 0
    assert len(result.primary_risk_drivers) > 0


def test_batch_prediction(test_encounter):
    enc2 = PatientEncounterInput(
        age="[40-50)",
        race="AfricanAmerican",
        gender="Male",
        time_in_hospital=2,
        number_inpatient=0,
        number_emergency=0,
        num_medications=8,
        number_diagnoses=4,
        A1Cresult="Norm",
        max_glu_serum="None",
        insulin="No",
        diag_1="428",
        change="No",
        diabetesMed="No",
    )

    batch_res = prediction_service.predict_batch([test_encounter, enc2], threshold=0.15)

    assert isinstance(batch_res, BatchPredictionResult)
    assert batch_res.total_patients == 2
    assert len(batch_res.predictions) == 2
    assert (batch_res.high_risk_count + batch_res.moderate_risk_count + batch_res.low_risk_count) == 2
