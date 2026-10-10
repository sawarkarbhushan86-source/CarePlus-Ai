"""
CARE-PULSE AI — Prediction Schemas
Schemas for single patient encounter risk inference, batch predictions, and explanations.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PatientEncounterInput(BaseModel):
    # Demographics
    race: str = Field(default="Caucasian", description="Reported racial background")
    gender: str = Field(default="Female", description="Biological sex (Female/Male)")
    age: str = Field(default="[60-70)", description="Age decade bracket, e.g., [60-70)")

    # Admission Details
    admission_type_id: int = Field(default=1, description="1=Emergency, 2=Urgent, 3=Elective, etc.")
    discharge_disposition_id: int = Field(default=1, description="1=Discharged to home, 3=SNF, 6=Home Health, etc.")
    admission_source_id: int = Field(default=7, description="7=Emergency Room, 1=Physician Referral, etc.")

    # Clinical Encounter Metrics
    time_in_hospital: int = Field(default=4, ge=1, le=14, description="Length of hospital stay in days (1-14)")
    num_lab_procedures: int = Field(default=45, ge=1, le=140, description="Total diagnostic lab tests performed")
    num_procedures: int = Field(default=1, ge=0, le=6, description="Number of non-lab procedural interventions (0-6)")
    num_medications: int = Field(default=16, ge=1, le=90, description="Number of distinct medications administered")
    number_diagnoses: int = Field(default=7, ge=1, le=16, description="Number of diagnoses entered on encounter")

    # Prior Hospital Utilization (Past 12 Months)
    number_outpatient: int = Field(default=0, ge=0, description="Outpatient visits in prior year")
    number_emergency: int = Field(default=0, ge=0, description="Emergency department visits in prior year")
    number_inpatient: int = Field(default=1, ge=0, description="Inpatient hospital admissions in prior year")

    # Diagnostic Codes (ICD-9 or Primary Class)
    diag_1: str = Field(default="250.02", description="Primary ICD-9 diagnosis code (e.g. 250 for diabetes, 428 for heart failure)")
    diag_2: str = Field(default="401", description="Secondary ICD-9 code")
    diag_3: str = Field(default="272", description="Tertiary ICD-9 code")

    # Glycemic / Lab Status
    max_glu_serum: str = Field(default="None", description="Glucose serum test result: None, Norm, >200, >300")
    A1Cresult: str = Field(default=">8", description="HbA1c test result: None, Norm, >7, >8")

    # Diabetes Medications
    metformin: str = Field(default="No", description="Metformin status: No, Steady, Up, Down")
    insulin: str = Field(default="Up", description="Insulin regimen status: No, Steady, Up, Down")
    glipizide: str = Field(default="No", description="Glipizide status: No, Steady, Up, Down")
    glyburide: str = Field(default="No", description="Glyburide status: No, Steady, Up, Down")
    pioglitazone: str = Field(default="No", description="Pioglitazone status: No, Steady, Up, Down")
    rosiglitazone: str = Field(default="No", description="Rosiglitazone status: No, Steady, Up, Down")
    change: str = Field(default="Ch", description="Medication changed during encounter: Ch (yes) or No")
    diabetesMed: str = Field(default="Yes", description="Any diabetic medication prescribed: Yes or No")

    @classmethod
    def validate_category(cls, val: str, allowed: List[str], default: str) -> str:
        if not val or not isinstance(val, str):
            return default
        cleaned = val.strip()
        for option in allowed:
            if cleaned.lower() == option.lower():
                return option
        return default


class FeatureContribution(BaseModel):
    feature: str
    feature_value: Any
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResult(BaseModel):
    readmission_probability: float
    predicted_class: int  # 0 or 1
    risk_tier: str  # "Low Risk", "Moderate Risk", "High Risk"
    decision_threshold: float
    confidence_interval_low: float
    confidence_interval_high: float
    odds_ratio_vs_baseline: float
    primary_risk_drivers: List[FeatureContribution]
    recommended_interventions: List[str]
    model_name: str
    clinical_disclaimer: str
    is_simulated_record: bool = True
    baseline_expected_value: Optional[float] = None
    positive_risk_contributions: Optional[float] = None
    negative_risk_contributions: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    encounters: List[PatientEncounterInput]
    decision_threshold: Optional[float] = 0.15


class BatchPredictionResult(BaseModel):
    total_patients: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResult]
