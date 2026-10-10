"""
CARE-PULSE AI — Model Evaluation and Prediction API Routes
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Body
import numpy as np
import joblib

from app.config import settings
from app.schemas.prediction import (
    PatientEncounterInput,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResult,
)
from app.schemas.experiment import ModelBenchmarkSummary
from app.services.training_service import training_service
from app.services.prediction_service import prediction_service
from app.services.calibration_service import calibration_service
from app.services.evaluation_service import evaluation_service
from app.services.subgroup_analysis import subgroup_analysis_service

router = APIRouter(prefix="/models", tags=["Model Lab & Predictions"])


@router.get("/benchmark", response_model=ModelBenchmarkSummary, summary="Retrieve Model Benchmark Results")
def get_benchmark_results():
    """Retrieve discrimination (ROC/PR), calibration (ECE/Brier), and sensitivity benchmarks."""
    summary = training_service.get_benchmark_summary()
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark summary not found. Please trigger model training via POST /api/v1/models/train first.",
        )
    return summary


@router.get("/curves", summary="ROC and PR Curve Coordinates for Charting")
def get_performance_curves(model_name: str = Query(default="Calibrated XGBoost")):
    """Fetch true positive rates, false positive rates, precision, and recall coordinates."""
    eval_cache_file = settings.METRICS_DIR / "test_evaluation_cache.joblib"
    if not eval_cache_file.exists():
        raise HTTPException(status_code=404, detail="Test evaluation cache not found. Please train models first.")

    cache_data = joblib.load(eval_cache_file)
    X_test = cache_data["X_test"]
    y_test = cache_data["y_test"].values

    pipeline = training_service.load_model(model_name)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return evaluation_service.get_curve_data(y_test, y_prob)


@router.get("/calibration", summary="Reliability Diagram Points and ECE")
def get_calibration_diagram(model_name: str = Query(default="Calibrated XGBoost")):
    """Compute calibration curve coordinates (observed vs predicted risk in 10 bins) and ECE."""
    eval_cache_file = settings.METRICS_DIR / "test_evaluation_cache.joblib"
    if not eval_cache_file.exists():
        raise HTTPException(status_code=404, detail="Test evaluation cache not found. Please train models first.")

    cache_data = joblib.load(eval_cache_file)
    X_test = cache_data["X_test"]
    y_test = cache_data["y_test"].values

    pipeline = training_service.load_model(model_name)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return calibration_service.compute_reliability_curve(y_test, y_prob, n_bins=10)


@router.get("/subgroups", summary="Demographic and Clinical Subgroup Fairness Audit")
def get_subgroup_fairness(
    model_name: str = Query(default="Calibrated XGBoost"),
    threshold: float = Query(default=0.15),
):
    """Audit true positive rate parity, false positive rates, and AUC across age, race, and gender."""
    eval_cache_file = settings.METRICS_DIR / "test_evaluation_cache.joblib"
    if not eval_cache_file.exists():
        raise HTTPException(status_code=404, detail="Test evaluation cache not found. Please train models first.")

    cache_data = joblib.load(eval_cache_file)
    X_test = cache_data["X_test"]
    y_test = cache_data["y_test"].values
    raw_test_df = cache_data.get("raw_test_df", X_test)

    pipeline = training_service.load_model(model_name)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return subgroup_analysis_service.get_fairness_audit(
        df_test=raw_test_df, y_true=y_test, y_prob=y_prob, threshold=threshold
    )


@router.post("/predict", response_model=PredictionResult, summary="Single Patient Readmission Risk Scoring")
def predict_patient(
    encounter: PatientEncounterInput,
    model_name: Optional[str] = Query(default="Calibrated XGBoost"),
    threshold: Optional[float] = Query(default=0.15),
):
    """Predict 30-day readmission risk, confidence interval, risk tier, local SHAP drivers, and care plan."""
    return prediction_service.predict_encounter(encounter, model_name=model_name, threshold=threshold)


@router.post("/predict-batch", response_model=BatchPredictionResult, summary="Cohort Batch Patient Risk Scoring")
def predict_batch_encounters(
    request: BatchPredictionRequest,
    model_name: Optional[str] = Query(default="Calibrated XGBoost"),
):
    """Score a batch cohort of hospital encounters."""
    return prediction_service.predict_batch(
        request.encounters, model_name=model_name, threshold=request.decision_threshold
    )


@router.post("/train", response_model=ModelBenchmarkSummary, summary="Trigger Autonomous Model Training Pipeline")
def trigger_training(
    subsample: Optional[int] = Query(default=25000, description="Cohort size for training speed (None for full)"),
    use_sample_data: bool = Query(default=False),
):
    """Execute end-to-end training of Logistic Regression, Random Forest, XGBoost, and Calibrated XGBoost."""
    summary = training_service.train_all_models(
        subsample_size=subsample, use_sample_data=use_sample_data
    )
    return summary
