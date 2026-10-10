"""
CARE-PULSE AI — Hospital Operations and Capacity Planning Service
Simulates hospital bed-day utilization, readmission prevention intervention economics,
and Return on Investment (ROI) for targeted nurse transition programs.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from app.config import settings
from app.schemas.report import OperationsImpactSummary
from app.services.training_service import training_service

logger = logging.getLogger(__name__)


class OperationsService:
    """Models healthcare resource allocation, bed capacity pressure, and clinical intervention ROI."""

    def __init__(self):
        self._test_probs: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None

    def _load_test_cache(self):
        """Lazily load test set probabilities and labels for empirical threshold modeling."""
        if self._test_probs is not None:
            return
        eval_cache_file = settings.METRICS_DIR / "test_evaluation_cache.joblib"
        if eval_cache_file.exists():
            try:
                import joblib
                cache_data = joblib.load(eval_cache_file)
                self._test_labels = cache_data["y_test"].values
                pipeline = training_service.load_model("Calibrated XGBoost")
                self._test_probs = pipeline.predict_proba(cache_data["X_test"])[:, 1]
            except Exception as e:
                logger.warning(f"Could not load test evaluation cache in OperationsService: {e}")

    def get_empirical_performance_at_threshold(self, threshold: float) -> Tuple[float, float]:
        """
        Derive empirical flagged percentage and positive predictive value (precision)
        at the specified decision threshold from held-out test cohort evaluations.
        """
        self._load_test_cache()
        if self._test_probs is not None and self._test_labels is not None:
            flagged = self._test_probs >= threshold
            flagged_rate = float(np.mean(flagged))
            precision = float(self._test_labels[flagged].mean()) if flagged.sum() > 0 else 0.0
            return max(0.01, min(0.95, flagged_rate)), max(0.05, min(0.95, precision))

        # Calibrated fallback model if cache not loaded
        # Empirical anchor: at threshold 0.15, flagged ~ 15.8%, precision ~ 23.9%
        flagged_rate = max(0.02, min(0.60, 0.158 * (0.15 / max(0.01, threshold)) ** 0.85))
        precision = max(0.08, min(0.45, 0.239 * (threshold / 0.15) ** 0.35))
        return flagged_rate, precision

    def simulate_intervention_program(
        self,
        cohort_size: int = 1000,
        decision_threshold: float = 0.15,
        avg_readmission_cost: float = 14500.0,
        intervention_cost_per_patient: float = 75.0,
        relative_risk_reduction: float = 0.20,
        baseline_readmission_rate: float = 0.1116,
        avg_length_of_stay_days: float = 4.68,
        intervention_uptake_rate: float = 1.0,
    ) -> OperationsImpactSummary:
        """
        Simulate financial and operational ROI of deploying predictive risk-targeted post-discharge care.
        Targeting patients above decision_threshold with a nurse transition program.
        
        Formulas:
        - High-Risk Patients = Cohort Size * Flagged Rate(threshold)
        - Enrolled Patients = High-Risk Patients * Uptake Rate
        - Expected Events in Enrolled = Enrolled Patients * Precision(threshold)
        - Prevented Readmissions = Expected Events in Enrolled * Relative Risk Reduction (RRR)
        - Program Cost = Enrolled Patients * Cost Per Patient
        - Gross Savings = Prevented Readmissions * Cost Per Readmission
        - Net Savings = Gross Savings - Program Cost
        - ROI (%) = (Net Savings / Program Cost) * 100 (safely handles zero cost)
        - Acute Bed-Days Freed = Prevented Readmissions * Average Length of Stay
        """
        if cohort_size <= 0:
            return OperationsImpactSummary(
                cohort_size=0,
                predicted_high_risk=0,
                predicted_moderate_risk=0,
                predicted_low_risk=0,
                baseline_expected_readmissions=0,
                prevented_readmissions_estimate=0,
                total_intervention_cost_usd=0.0,
                gross_readmission_savings_usd=0.0,
                net_savings_usd=0.0,
                estimated_roi_percentage=0.0,
                beds_freed_estimate=0,
            )

        flagged_fraction, precision = self.get_empirical_performance_at_threshold(decision_threshold)
        moderate_risk_fraction = min(0.30, max(0.05, 0.35 - flagged_fraction))
        low_risk_fraction = max(0.0, 1.0 - flagged_fraction - moderate_risk_fraction)

        high_risk_patients = int(round(cohort_size * flagged_fraction))
        mod_risk_patients = int(round(cohort_size * moderate_risk_fraction))
        low_risk_patients = max(0, cohort_size - high_risk_patients - mod_risk_patients)

        # Baseline expected readmissions without intervention
        baseline_readmissions = int(round(cohort_size * baseline_readmission_rate))

        # Enrolled patients (target group adjusted for consent/uptake rate)
        enrolled_patients = int(round(high_risk_patients * intervention_uptake_rate))

        # Events in enrolled group based on model precision (positive predictive value)
        expected_events_in_enrolled = enrolled_patients * precision

        # Prevented readmissions based on interventional trial relative risk reduction
        prevented_readmissions = int(round(expected_events_in_enrolled * relative_risk_reduction))

        # Financial Calculations
        total_intervention_cost = enrolled_patients * intervention_cost_per_patient
        gross_savings = prevented_readmissions * avg_readmission_cost
        net_savings = gross_savings - total_intervention_cost

        # Safe ROI calculation avoiding division by zero
        if total_intervention_cost > 0:
            roi_percentage = (net_savings / total_intervention_cost) * 100.0
        elif gross_savings > 0:
            roi_percentage = 999.9  # Infinitely positive return (costless intervention)
        else:
            roi_percentage = 0.0

        # Bed Capacity impact based on empirical average length of stay
        beds_freed_days = int(round(prevented_readmissions * avg_length_of_stay_days))

        return OperationsImpactSummary(
            cohort_size=cohort_size,
            predicted_high_risk=high_risk_patients,
            predicted_moderate_risk=mod_risk_patients,
            predicted_low_risk=low_risk_patients,
            baseline_expected_readmissions=baseline_readmissions,
            prevented_readmissions_estimate=prevented_readmissions,
            total_intervention_cost_usd=round(total_intervention_cost, 2),
            gross_readmission_savings_usd=round(gross_savings, 2),
            net_savings_usd=round(net_savings, 2),
            estimated_roi_percentage=round(roi_percentage, 1),
            beds_freed_estimate=beds_freed_days,
        )

    def calculate_break_even_effectiveness(
        self,
        decision_threshold: float = 0.15,
        intervention_cost_per_patient: float = 75.0,
        avg_readmission_cost: float = 14500.0,
        cohort_size: int = 1000,
        intervention_uptake_rate: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Compute minimum required interventional Relative Risk Reduction (RRR)
        to achieve financial break-even (Net Savings = $0, ROI = 0%).
        
        Formula:
        RRR_break_even = Intervention_Cost / (Precision(threshold) * Readmission_Cost)
        """
        _, precision = self.get_empirical_performance_at_threshold(decision_threshold)
        flagged_rate, _ = self.get_empirical_performance_at_threshold(decision_threshold)

        if precision <= 0 or avg_readmission_cost <= 0:
            return {
                "break_even_rrr_percentage": 0.0,
                "break_even_prevented_readmissions": 0,
                "precision_at_threshold": round(precision, 4),
                "is_viable": False,
            }

        break_even_rrr = intervention_cost_per_patient / (precision * avg_readmission_cost)
        enrolled = int(round(cohort_size * flagged_rate * intervention_uptake_rate))
        total_cost = enrolled * intervention_cost_per_patient
        break_even_prevented = int(math.ceil(total_cost / avg_readmission_cost)) if avg_readmission_cost > 0 else 0

        return {
            "break_even_rrr_percentage": round(break_even_rrr * 100.0, 2),
            "break_even_prevented_readmissions": break_even_prevented,
            "precision_at_threshold": round(precision, 4),
            "total_enrolled_patients": enrolled,
            "total_program_cost": round(total_cost, 2),
            "is_viable": bool(break_even_rrr < 0.50),
        }

    def calculate_bed_capacity_forecast(
        self,
        monthly_discharges: int = 1200,
        baseline_readmission_rate: float = 0.1116,
        avg_readmission_los_days: float = 4.5,
        total_licensed_beds: int = 350,
        peak_surge_factor: float = 1.20,
    ) -> Dict[str, Any]:
        """Model average and peak bed occupancy pressure generated by 30-day diabetic readmissions."""
        monthly_readmissions = int(round(monthly_discharges * baseline_readmission_rate))
        annual_readmissions = monthly_readmissions * 12
        total_readmission_bed_days_monthly = round(monthly_readmissions * avg_readmission_los_days, 1)
        total_readmission_bed_days_annual = round(annual_readmissions * avg_readmission_los_days, 1)

        average_beds_occupied = round(total_readmission_bed_days_monthly / 30.0, 1)
        average_bed_capacity_share_pct = round((average_beds_occupied / max(1, total_licensed_beds)) * 100.0, 1)

        peak_beds_demand = round(average_beds_occupied * peak_surge_factor, 1)
        peak_bed_capacity_share_pct = round((peak_beds_demand / max(1, total_licensed_beds)) * 100.0, 1)

        return {
            "monthly_discharges": monthly_discharges,
            "annual_discharges": monthly_discharges * 12,
            "projected_monthly_readmissions": monthly_readmissions,
            "projected_annual_readmissions": annual_readmissions,
            "monthly_readmission_bed_days": total_readmission_bed_days_monthly,
            "projected_readmission_bed_days": total_readmission_bed_days_monthly,
            "annual_readmission_bed_days": total_readmission_bed_days_annual,
            "total_licensed_beds": total_licensed_beds,
            "average_daily_beds_occupied": average_beds_occupied,
            "equivalent_beds_continuously_occupied": average_beds_occupied,
            "average_bed_capacity_share_pct": average_bed_capacity_share_pct,
            "readmission_bed_capacity_share_pct": average_bed_capacity_share_pct,
            "peak_surge_factor": peak_surge_factor,
            "peak_bed_demand": peak_beds_demand,
            "peak_bed_capacity_share_pct": peak_bed_capacity_share_pct,
            "surge_assumptions": f"Peak bed demand assumes a {peak_surge_factor:.2f}x surge multiplier accounting for midweek admission spikes and seasonal respiratory census peaks.",
        }


operations_service = OperationsService()
