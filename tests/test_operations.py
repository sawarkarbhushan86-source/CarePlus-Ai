"""
Unit tests for CARE-PULSE AI Hospital Operations and Capacity Planning Service.
"""

import pytest
from app.services.operations_service import operations_service
from app.schemas.report import OperationsImpactSummary


def test_intervention_program_simulation():
    res = operations_service.simulate_intervention_program(
        cohort_size=1000,
        decision_threshold=0.15,
        avg_readmission_cost=14500.0,
        intervention_cost_per_patient=75.0,
    )

    assert isinstance(res, OperationsImpactSummary)
    assert res.cohort_size == 1000
    assert res.predicted_high_risk > 0
    assert res.prevented_readmissions_estimate > 0
    assert res.total_intervention_cost_usd == res.predicted_high_risk * 75.0
    assert res.gross_readmission_savings_usd == res.prevented_readmissions_estimate * 14500.0
    assert res.net_savings_usd == res.gross_readmission_savings_usd - res.total_intervention_cost_usd
    assert res.estimated_roi_percentage > 0.0
    assert res.beds_freed_estimate > 0


def test_bed_capacity_forecast():
    forecast = operations_service.calculate_bed_capacity_forecast(
        monthly_discharges=1200,
        total_licensed_beds=350,
        avg_readmission_los_days=4.5,
    )

    assert forecast["monthly_discharges"] == 1200
    assert forecast["projected_monthly_readmissions"] > 0
    assert forecast["projected_readmission_bed_days"] > 0
    assert forecast["total_licensed_beds"] == 350
    assert 0.0 < forecast["readmission_bed_capacity_share_pct"] < 100.0
