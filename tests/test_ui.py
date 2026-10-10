"""
CARE-PULSE AI — UI/UX and Navigation Automated Test Suite
Validates application imports, API client reliability, in-process fallbacks,
8-view page rendering, decision thresholds, and regression tests against blank screen issues.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ui.api_client import api_client, APIClient
from ui.views import (
    overview,
    data_explorer,
    statistical_analysis,
    model_lab,
    explainability,
    reliability,
    hospital_analytics,
    reports,
)
from app.schemas.prediction import PatientEncounterInput
from app.schemas.report import ReportGenerationRequest


# 1. Application Imports
def test_ui_module_imports():
    """Verify all UI views and client modules import cleanly without circular dependencies."""
    from ui import dashboard, api_client, components
    assert hasattr(dashboard, "pages")
    assert hasattr(api_client, "api_client")
    assert hasattr(components, "render_header")


# 2. API Client Success and Failure Paths
def test_api_client_health_online():
    """Test API client health check when backend is online."""
    health = api_client.get_health()
    assert "status" in health
    assert "champion_model" in health
    assert health["dataset_available"] is True
    assert health["models_trained"] is True


def test_api_client_offline_fallback():
    """Verify API client gracefully falls back to direct in-process services without crashing."""
    offline_client = APIClient(base_url="http://127.0.0.1:9999/api/v1")
    assert offline_client.is_api_alive() is False

    health = offline_client.get_health()
    assert health["status"] in ["healthy", "ready_for_setup"]
    assert health["backend_connected"] is False

    summary = offline_client.get_dataset_summary()
    assert summary.get("total_rows") == 101766

    benchmark = offline_client.get_benchmark_summary()
    assert benchmark is not None
    assert "Calibrated XGBoost" in benchmark["models_evaluated"]


# 3. 8-View Page Rendering Verification
@pytest.mark.parametrize(
    "view_name,view_func",
    [
        ("Overview", overview.render),
        ("Data Explorer", data_explorer.render),
        ("Statistical Analysis", statistical_analysis.render),
        ("Model Lab", model_lab.render),
        ("Explainability", explainability.render),
        ("Reliability & Fairness", reliability.render),
        ("Hospital Analytics", hospital_analytics.render),
        ("Reports", reports.render),
    ],
)
def test_view_render_execution(view_name, view_func):
    """Verify each of the 8 clinical analytics views executes without uncaught exceptions."""
    try:
        view_func()
    except Exception as e:
        pytest.fail(f"View {view_name} raised uncaught exception during render: {e}")


# 4. Data Quality Calculations
def test_data_quality_report():
    """Verify dataset validation audit calculations."""
    report = api_client.get_validation_report()
    assert report["total_rows"] >= 5000
    assert "missingness_summary" in report
    assert "target_distribution" in report


# 5. Statistical Calculations
def test_odds_ratios_calculation():
    """Verify biostatistical Odds Ratio calculations and confidence intervals."""
    ors = api_client.get_odds_ratios()
    assert len(ors) > 0
    for item in ors:
        assert item["odds_ratio"] > 0
        assert item["ci_95_low"] <= item["ci_95_high"]
        assert 0.0 <= item["p_value"] <= 1.0


# 6. Model Metrics and Decision Threshold Calculations
def test_threshold_sensitivity_calculation():
    """Verify threshold slider adjustments produce valid clinical metrics."""
    from app.services.evaluation_service import evaluation_service
    y_test = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.05, 0.40, 0.12, 0.25, 0.08, 0.10, 0.60, 0.14])

    m15 = evaluation_service.evaluate_model("Test", y_test, y_prob, threshold=0.15)
    m30 = evaluation_service.evaluate_model("Test", y_test, y_prob, threshold=0.30)

    # Lower threshold should yield higher or equal recall
    assert m15.recall >= m30.recall
    # Higher threshold should yield higher or equal specificity
    assert m30.specificity >= m15.specificity


# 7. Prediction and Explanation Handling
def test_single_encounter_prediction():
    """Verify patient simulation risk scoring and care plan generation."""
    enc = PatientEncounterInput(
        age="[70-80)",
        race="AfricanAmerican",
        gender="Female",
        time_in_hospital=6,
        number_inpatient=2,
        number_emergency=1,
        num_medications=18,
        number_diagnoses=8,
        diag_1="428",
        A1Cresult=">8",
        insulin="Up",
    )
    res = api_client.predict_encounter(enc, model_name="Calibrated XGBoost", threshold=0.15)
    assert 0.0 <= res["readmission_probability"] <= 1.0
    assert res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert len(res["primary_risk_drivers"]) > 0
    assert len(res["recommended_interventions"]) > 0


# 8. Report Generation
def test_report_generation_all_types():
    """Verify all 3 report types generate markdown documentation successfully."""
    for rep_type in ["clinical_audit", "executive_summary", "operations_plan"]:
        req = ReportGenerationRequest(
            report_type=rep_type,
            decision_threshold=0.15,
            planned_encounters_count=1000,
        )
        res = api_client.generate_report(req)
        assert res["report_type"] == rep_type
        assert len(res["markdown_content"]) > 100
        assert res["report_id"].startswith("CPR-")


# 9. Hospital Operations Simulation
def test_operations_simulation():
    """Verify hospital care transition program ROI calculations."""
    sim = api_client.simulate_operations(
        cohort_size=2000,
        decision_threshold=0.15,
        nurse_cost=75.0,
        readmission_cost=14500.0,
    )
    assert sim["cohort_size"] == 2000
    assert sim["predicted_high_risk"] > 0
    assert sim["prevented_readmissions_estimate"] > 0
    assert sim["net_savings_usd"] > 0
    assert sim["estimated_roi_percentage"] > 0


# 10. Regression Test for Blank Screen
def test_regression_no_blank_screen():
    """Regression test ensuring dashboard definition contains 8 registered pages and runs cleanly."""
    from ui.dashboard import pages
    assert len(pages) == 8

    # Verify dashboard execution inside Streamlit AppTest runner
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    dashboard_file = Path(__file__).resolve().parent.parent / "ui" / "dashboard.py"
    at = AppTest.from_file(str(dashboard_file), default_timeout=15).run()
    assert len(at.exception) == 0
    assert len(at.metric) >= 4
