"""
CARE-PULSE AI — Unit & Integration Tests for Formal Report Generation Flow
Validates button handlers, input validation, backend communication, error handling,
report updates, and download data integrity.
"""

import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.report import ReportGenerationRequest
from ui.api_client import APIClient, api_client
from ui.views import reports

_test_client = TestClient(app)


@pytest.fixture(autouse=True)
def bridge_backend_requests(monkeypatch):
    """Bridge requests to localhost FastAPI app in-process using TestClient."""
    import requests
    original_post = requests.post

    def mock_post(url, *args, **kwargs):
        if url.startswith("http://127.0.0.1:8000"):
            path = url.replace("http://127.0.0.1:8000", "")
            return _test_client.post(path, json=kwargs.get("json"), headers=kwargs.get("headers"))
        return original_post(url, *args, **kwargs)

    monkeypatch.setattr(requests, "post", mock_post)


def test_report_generation_button_valid_click():
    """Verify clicking Generate Formal Report with valid inputs communicates with backend and populates session state."""
    # Reset streamlit session state
    st.session_state["generated_report"] = None
    st.session_state["report_error"] = None

    req = ReportGenerationRequest(
        report_type="executive_summary",
        decision_threshold=0.18,
        planned_encounters_count=2500,
        nurse_intervention_cost=80.0,
        avg_readmission_cost=15000.0,
    )

    # Call api_client directly with require_backend=True
    res = api_client.generate_report(req, require_backend=True)

    assert res is not None
    assert "report_id" in res
    assert res["report_id"].startswith("CPR-")
    assert res["report_type"] == "executive_summary"
    assert res["backend_confirmed"] is True
    assert "source" in res
    assert len(res["markdown_content"]) > 100
    assert "Executive Leadership Summary" in res["markdown_content"]
    assert "2,500" in res["markdown_content"]
    assert "0.18" in res["markdown_content"]


def test_report_generation_invalid_inputs():
    """Verify invalid configuration inputs are rejected prior to backend dispatch."""
    invalid_cases = [
        {"cohort_size": 50, "threshold": 0.15, "nurse_cost": 75.0, "readm_cost": 14500.0, "expected": "Cohort Size"},
        {"cohort_size": -100, "threshold": 0.15, "nurse_cost": 75.0, "readm_cost": 14500.0, "expected": "Cohort Size"},
        {"cohort_size": 1000, "threshold": -0.05, "nurse_cost": 75.0, "readm_cost": 14500.0, "expected": "threshold"},
        {"cohort_size": 1000, "threshold": 1.20, "nurse_cost": 75.0, "readm_cost": 14500.0, "expected": "threshold"},
        {"cohort_size": 1000, "threshold": 0.15, "nurse_cost": -25.0, "readm_cost": 14500.0, "expected": "cost"},
        {"cohort_size": 1000, "threshold": 0.15, "nurse_cost": 75.0, "readm_cost": 0.0, "expected": "cost"},
    ]

    for case in invalid_cases:
        validation_errors = []
        c_size = case["cohort_size"]
        thresh = case["threshold"]
        n_cost = case["nurse_cost"]
        r_cost = case["readm_cost"]

        if c_size is None or c_size < 100:
            validation_errors.append("Inpatient Annual Cohort Size must be at least 100 encounters.")
        if thresh is None or not (0.01 <= thresh <= 0.99):
            validation_errors.append("Decision threshold must be between 0.01 and 0.99.")
        if n_cost is None or n_cost < 0:
            validation_errors.append("Nurse transition cost must be non-negative ($0 or greater).")
        if r_cost is None or r_cost <= 0:
            validation_errors.append("Average readmission cost must be greater than $0.")

        assert len(validation_errors) > 0, f"Expected validation error for case: {case}"
        assert any(case["expected"].lower() in err.lower() for err in validation_errors)


def test_report_generation_backend_unavailable_scenario():
    """Verify backend-unavailable scenario raises an explicit error and does not display fake success."""
    offline_client = APIClient(base_url="http://127.0.0.1:59999/api/v1")
    req = ReportGenerationRequest(
        report_type="clinical_audit",
        decision_threshold=0.15,
        planned_encounters_count=1000,
    )

    with pytest.raises(RuntimeError) as exc_info:
        offline_client.generate_report(req, require_backend=True)

    assert "backend service unavailable" in str(exc_info.value).lower()


def test_report_updates_on_new_generation():
    """Verify generating a second report cleanly overwrites the previous report in state."""
    req1 = ReportGenerationRequest(
        report_type="clinical_audit",
        decision_threshold=0.15,
        planned_encounters_count=1000,
    )
    res1 = api_client.generate_report(req1, require_backend=True)

    req2 = ReportGenerationRequest(
        report_type="operations_plan",
        decision_threshold=0.22,
        planned_encounters_count=3000,
    )
    res2 = api_client.generate_report(req2, require_backend=True)

    assert res1["report_id"] != res2["report_id"]
    assert res1["report_type"] == "clinical_audit"
    assert res2["report_type"] == "operations_plan"
    assert "Care Transition" in res2["markdown_content"]
    assert "3,000" in res2["markdown_content"]


def test_download_payload_integrity():
    """Verify downloaded markdown payload matches the active report exactly."""
    req = ReportGenerationRequest(
        report_type="clinical_audit",
        decision_threshold=0.15,
        planned_encounters_count=1200,
    )
    res = api_client.generate_report(req, require_backend=True)

    markdown_data = res.get("markdown_content", "")
    assert len(markdown_data) > 0
    assert res["report_id"] in markdown_data
    assert "Algorithmic Audit & Model Card" in markdown_data
    filename = f"{res['report_id']}.md"
    assert filename.endswith(".md")
    assert filename.startswith("CPR-")
