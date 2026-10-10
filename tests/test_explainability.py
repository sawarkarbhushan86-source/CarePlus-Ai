"""
Unit tests for CARE-PULSE AI Explainability Service.
"""

import pytest
import pandas as pd
from app.services.explainability_service import explainability_service, clean_feature_label
from app.schemas.prediction import PatientEncounterInput
from app.services.training_service import training_service
from app.services.preprocessing import preprocessing_service


def test_clean_feature_label():
    assert clean_feature_label("num__number_inpatient") == "Prior Inpatient Admissions"
    assert clean_feature_label("cat__A1Cresult_>8") == "HbA1c Result > 8%"
    assert clean_feature_label("cat__insulin_Up") == "Insulin Dose Increased"
    assert clean_feature_label("cat__diag_1_group_Circulatory") == "Primary Diagnosis: Circulatory System"


def test_global_feature_importance():
    res = explainability_service.compute_global_feature_importance(model_name="XGBoost", top_n=10)
    assert "features" in res
    assert "mean_abs_shap" in res
    assert len(res["features"]) == len(res["mean_abs_shap"])
    assert len(res["features"]) >= 5


def test_local_patient_explanation():
    enc = PatientEncounterInput(
        age="[70-80)",
        race="Caucasian",
        gender="Male",
        time_in_hospital=5,
        number_inpatient=2,
        number_emergency=1,
        num_medications=18,
        number_diagnoses=8,
        A1Cresult=">8",
        max_glu_serum="None",
        insulin="Up",
        diag_1="250.02",
        change="Ch",
        diabetesMed="Yes",
    )

    pipeline = training_service.load_model("Calibrated XGBoost")
    df_input = pd.DataFrame([enc.dict()])
    df_eng = preprocessing_service.engineer_features(df_input)

    contributions = explainability_service.explain_patient_encounter(
        pipeline=pipeline, patient_df=df_eng, top_k=5
    )

    assert len(contributions) > 0
    assert contributions[0].direction in ["increases_risk", "decreases_risk"]
    assert isinstance(contributions[0].shap_value, float)
