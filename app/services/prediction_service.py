"""
CARE-PULSE AI — Real-time Prediction Service
Provides patient-level 30-day readmission risk inference, confidence intervals,
risk tier categorization, local SHAP explanation drivers, and evidence-based clinical recommendations.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from app.config import settings
from app.schemas.prediction import (
    PatientEncounterInput,
    PredictionResult,
    BatchPredictionResult,
    FeatureContribution,
)
from app.services.training_service import training_service
from app.services.preprocessing import preprocessing_service
from app.services.explainability_service import explainability_service
from app.services.uncertainty_service import uncertainty_service

logger = logging.getLogger(__name__)

CLINICAL_DISCLAIMER = (
    "RESEARCH & PORTFOLIO DEMONSTRATION ONLY: "
    "CARE-PULSE AI is an analytical research tool and not a cleared clinical decision support system. "
    "Do not use for diagnosis, treatment decisions, or autonomous patient discharge."
)


class PredictionService:
    """Handles single-encounter and cohort batch risk inference."""

    def __init__(self):
        self._active_model_name = "Calibrated XGBoost"

    def predict_encounter(
        self,
        encounter: PatientEncounterInput,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Score a single patient encounter and produce explanations and clinical recommendations."""
        target_model = model_name or self._active_model_name
        decision_thresh = threshold if threshold is not None else settings.DEFAULT_RISK_THRESHOLD

        # Load model pipeline
        try:
            pipeline = training_service.load_model(target_model)
        except Exception:
            try:
                pipeline = training_service.load_model("XGBoost")
                target_model = "XGBoost"
            except Exception:
                pipeline = training_service.load_model("Logistic Regression")
                target_model = "Logistic Regression"

        # Convert input Pydantic model to dict and sanitize categories
        encounter_dict = encounter.model_dump()
        
        # Categorical normalization and fallback handling
        encounter_dict["race"] = encounter.validate_category(
            encounter_dict.get("race", ""),
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
            "Other",
        )
        encounter_dict["gender"] = encounter.validate_category(
            encounter_dict.get("gender", ""),
            ["Female", "Male"],
            "Female",
        )
        encounter_dict["age"] = encounter.validate_category(
            encounter_dict.get("age", ""),
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            "[60-70)",
        )
        encounter_dict["A1Cresult"] = encounter.validate_category(
            encounter_dict.get("A1Cresult", ""),
            [">8", ">7", "Norm", "None"],
            "None",
        )
        encounter_dict["max_glu_serum"] = encounter.validate_category(
            encounter_dict.get("max_glu_serum", ""),
            [">300", ">200", "Norm", "None"],
            "None",
        )
        encounter_dict["insulin"] = encounter.validate_category(
            encounter_dict.get("insulin", ""),
            ["Up", "Down", "Steady", "No"],
            "No",
        )
        encounter_dict["change"] = encounter.validate_category(
            encounter_dict.get("change", ""),
            ["Ch", "No"],
            "No",
        )
        encounter_dict["diabetesMed"] = encounter.validate_category(
            encounter_dict.get("diabetesMed", ""),
            ["Yes", "No"],
            "No",
        )

        df_input = pd.DataFrame([encounter_dict])

        # Feature engineering
        df_engineered = preprocessing_service.engineer_features(df_input)

        # Predict probability
        try:
            y_prob = float(pipeline.predict_proba(df_engineered)[:, 1][0])
        except Exception as e:
            logger.warning(f"Error predicting directly on engineered frame: {e}")
            # Ensure categorical IDs are strings
            for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
                if col in df_engineered.columns:
                    df_engineered[col] = df_engineered[col].astype(str)
            y_prob = float(pipeline.predict_proba(df_engineered)[:, 1][0])

        # Risk tiering
        if y_prob < settings.LOW_RISK_CUTOFF:
            risk_tier = "Low Risk"
        elif y_prob < settings.HIGH_RISK_CUTOFF:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "High Risk"

        predicted_class = 1 if y_prob >= decision_thresh else 0

        # Confidence Interval
        complexity = (
            int(encounter.number_diagnoses > 7)
            + int(encounter.number_inpatient > 0)
            + int(encounter.num_medications > 15)
        )
        ci_low, ci_high, _ = uncertainty_service.estimate_probability_ci(y_prob, encounter_complexity=complexity)

        # Baseline odds ratio (relative to ~11% average cohort readmission rate)
        baseline_rate = 0.1116
        baseline_odds = baseline_rate / (1.0 - baseline_rate)
        patient_odds = y_prob / (1.0 - y_prob) if y_prob < 1.0 else 10.0
        odds_ratio = round(patient_odds / baseline_odds, 2)

        # Local SHAP contributions
        drivers = explainability_service.explain_patient_encounter(
            pipeline=pipeline, patient_df=df_engineered, top_k=6
        )

        pos_contrib = round(float(sum(d.shap_value for d in drivers if d.shap_value > 0)), 4)
        neg_contrib = round(float(sum(d.shap_value for d in drivers if d.shap_value < 0)), 4)
        base_log_odds = round(float(np.log(baseline_rate / (1.0 - baseline_rate))), 4)

        # Generate non-prescriptive educational observations
        recommendations = self._generate_recommendations(encounter, y_prob, drivers)

        return PredictionResult(
            readmission_probability=round(y_prob, 4),
            predicted_class=predicted_class,
            risk_tier=risk_tier,
            decision_threshold=decision_thresh,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            odds_ratio_vs_baseline=odds_ratio,
            primary_risk_drivers=drivers,
            recommended_interventions=recommendations,
            model_name=target_model,
            clinical_disclaimer=CLINICAL_DISCLAIMER,
            is_simulated_record=True,
            baseline_expected_value=base_log_odds,
            positive_risk_contributions=pos_contrib,
            negative_risk_contributions=neg_contrib,
        )

    def predict_batch(
        self,
        encounters: List[PatientEncounterInput],
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> BatchPredictionResult:
        """Score a cohort batch of patient encounters."""
        results = [self.predict_encounter(enc, model_name, threshold) for enc in encounters]

        high_count = sum(1 for r in results if r.risk_tier == "High Risk")
        mod_count = sum(1 for r in results if r.risk_tier == "Moderate Risk")
        low_count = sum(1 for r in results if r.risk_tier == "Low Risk")

        return BatchPredictionResult(
            total_patients=len(results),
            high_risk_count=high_count,
            moderate_risk_count=mod_count,
            low_risk_count=low_count,
            predictions=results,
        )

    def _generate_recommendations(
        self,
        encounter: PatientEncounterInput,
        prob: float,
        drivers: List[FeatureContribution],
    ) -> List[str]:
        """
        Formulate educational and analytical risk context based on model features.
        Strictly research-oriented: non-prescriptive, no clinical orders or treatment directives.
        """
        observations = []

        # Risk Stratification Context
        tier_text = "Elevated Risk" if prob >= 0.25 else ("Moderate Risk" if prob >= 0.10 else "Lower Risk")
        observations.append(
            f"Simulated Stratification Tier: {tier_text} ({prob * 100:.1f}% estimated 30-day readmission risk based on historical data patterns)."
        )

        # Associative Risk Factors (Non-causal)
        if encounter.number_inpatient >= 1:
            observations.append(
                f"Historical Utilization Association: Record indicates {encounter.number_inpatient} prior inpatient admission(s) "
                f"in preceding 12 months, historically correlating with recurrent hospitalization."
            )

        if encounter.insulin in ["Up", "Down", "Steady"] or encounter.change == "Ch":
            observations.append(
                "Pharmacotherapy Adjustment Flag: Inpatient diabetes medication modification or insulin titration "
                "noted, associated with transitional glycemic management needs."
            )

        if encounter.A1Cresult in [">7", ">8"]:
            observations.append(
                f"Glycemic Marker: Elevated HbA1c result ({encounter.A1Cresult}) indicates pre-admission chronic hyperglycemia."
            )
        elif encounter.A1Cresult == "None":
            observations.append(
                "Laboratory Data Gap: HbA1c was unrecorded during this encounter, limiting glycemic risk assessment."
            )

        if encounter.num_medications >= 15:
            observations.append(
                f"Medication Regimen Complexity: Patient receives {encounter.num_medications} distinct medications, "
                "a marker of multimorbidity and potential post-discharge transition complexity."
            )

        if encounter.time_in_hospital >= 7 or encounter.number_diagnoses >= 8:
            observations.append(
                f"Clinical Complexity Factor: Inpatient stay of {encounter.time_in_hospital} days "
                f"and {encounter.number_diagnoses} diagnoses reflect substantial acute multimorbidity."
            )

        # Mandatory Human Review & Limitations Notice
        observations.append(
            "MANDATORY CLINICAL REVIEW: This statistical risk score is an educational simulation based on historical "
            "records (1999–2008). It does NOT constitute a diagnosis, treatment plan, or discharge directive. "
            "Individual patient discharge decisions require comprehensive clinical assessment by qualified healthcare professionals."
        )

        return observations


prediction_service = PredictionService()
