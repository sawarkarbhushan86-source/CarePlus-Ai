"""
CARE-PULSE AI — Clinical Audit and Reporting API Routes
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Body
from app.services.reporting_service import reporting_service
from app.services.operations_service import operations_service
from app.schemas.report import (
    ReportGenerationRequest,
    GeneratedReportResponse,
    OperationsImpactSummary,
)

router = APIRouter(prefix="/reports", tags=["Reporting & Operations"])


@router.post("/generate", response_model=GeneratedReportResponse, summary="Generate Executive / Clinical Audit Report")
def generate_report(request: ReportGenerationRequest):
    """Compile comprehensive algorithmic audit, operational savings, and markdown documentation."""
    return reporting_service.generate_report(request)


@router.get("/operations-simulation", response_model=OperationsImpactSummary, summary="Simulate Clinical Intervention ROI")
def simulate_operations(
    cohort_size: int = Query(default=1000, ge=100),
    decision_threshold: float = Query(default=0.15, ge=0.05, le=0.50),
    nurse_cost: float = Query(default=75.0, ge=10.0),
    readmission_cost: float = Query(default=14500.0, ge=1000.0),
    uptake_rate: float = Query(default=0.80, ge=0.10, le=1.0),
    relative_risk_reduction: float = Query(default=0.20, ge=0.05, le=0.60),
):
    """Simulate targeted nurse care navigation program net savings, bed-days freed, and ROI percentage."""
    return operations_service.simulate_intervention_program(
        cohort_size=cohort_size,
        decision_threshold=decision_threshold,
        intervention_cost_per_patient=nurse_cost,
        avg_readmission_cost=readmission_cost,
        intervention_uptake_rate=uptake_rate,
        relative_risk_reduction=relative_risk_reduction,
    )


@router.get("/bed-capacity", summary="Inpatient Bed Occupancy Forecast")
def get_bed_capacity_forecast(
    monthly_discharges: int = Query(default=1200, ge=100),
    licensed_beds: int = Query(default=350, ge=50),
):
    """Forecast hospital bed pressure and capacity shares consumed by 30-day diabetic readmissions."""
    return operations_service.calculate_bed_capacity_forecast(
        monthly_discharges=monthly_discharges, total_licensed_beds=licensed_beds
    )
