"""
CARE-PULSE AI — Model Explainability API Routes
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from app.services.explainability_service import explainability_service
from app.services.prediction_service import prediction_service
from app.schemas.prediction import PatientEncounterInput, FeatureContribution

router = APIRouter(prefix="/explanations", tags=["Model Explainability & SHAP"])


@router.get("/global", summary="Population-wide SHAP Feature Importance Rankings")
def get_global_explanations(
    model_name: str = Query(default="XGBoost"),
    top_n: int = Query(default=20, ge=5, le=50),
):
    """Retrieve mean absolute SHAP values across all patient encounters for the specified model."""
    return explainability_service.compute_global_feature_importance(model_name=model_name, top_n=top_n)


@router.post("/local", response_model=List[FeatureContribution], summary="Local Patient-Level SHAP Waterfall Drivers")
def get_local_explanation(
    encounter: PatientEncounterInput,
    model_name: Optional[str] = Query(default="Calibrated XGBoost"),
    top_k: int = Query(default=8, ge=3, le=20),
):
    """Compute local SHAP contributions indicating exact clinical drivers of readmission risk for a patient."""
    result = prediction_service.predict_encounter(encounter, model_name=model_name)
    return result.primary_risk_drivers[:top_k]
