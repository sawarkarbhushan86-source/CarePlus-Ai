"""
CARE-PULSE AI — Report Schemas
Pydantic schemas for clinical audits, executive summaries, and operations planning.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ReportGenerationRequest(BaseModel):
    report_type: str = Field(default="clinical_audit", description="clinical_audit, executive_summary, or operations_plan")
    decision_threshold: float = Field(default=0.15)
    planned_encounters_count: int = Field(default=1000)
    nurse_intervention_cost: float = Field(default=75.0)
    avg_readmission_cost: float = Field(default=14500.0)


class OperationsImpactSummary(BaseModel):
    cohort_size: int
    predicted_high_risk: int
    predicted_moderate_risk: int
    predicted_low_risk: int
    baseline_expected_readmissions: int
    prevented_readmissions_estimate: int
    total_intervention_cost_usd: float
    gross_readmission_savings_usd: float
    net_savings_usd: float
    estimated_roi_percentage: float
    beds_freed_estimate: int


class GeneratedReportResponse(BaseModel):
    report_id: str
    report_type: str
    generated_at: str
    executive_summary: str
    operations_impact: OperationsImpactSummary
    recommendations: List[str]
    model_audit: Dict[str, Any]
    markdown_content: str
