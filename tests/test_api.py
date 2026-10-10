"""
Unit and integration tests for CARE-PULSE AI FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "CARE-PULSE AI"
    assert "documentation" in data


def test_health_endpoints():
    r1 = client.get("/api/v1/health")
    assert r1.status_code == 200
    assert r1.json()["app_name"] == "CARE-PULSE AI"

    r2 = client.get("/api/v1/health/system")
    assert r2.status_code == 200
    assert "python_version" in r2.json()


def test_dataset_summary_endpoint():
    response = client.get("/api/v1/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "columns" in data


def test_analytics_overview_endpoint():
    response = client.get("/api/v1/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "readmission_30d_prevalence" in data
    assert "avg_length_of_stay" in data


def test_models_benchmark_endpoint():
    response = client.get("/api/v1/models/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "best_model_name" in data
    assert "metrics" in data


def test_single_prediction_endpoint():
    payload = {
        "age": "[60-70)",
        "race": "Caucasian",
        "gender": "Female",
        "time_in_hospital": 4,
        "number_inpatient": 1,
        "number_emergency": 0,
        "num_medications": 16,
        "number_diagnoses": 7,
        "A1Cresult": ">8",
        "max_glu_serum": "None",
        "insulin": "Up",
        "diag_1": "250.02",
        "change": "Ch",
        "diabetesMed": "Yes",
    }
    response = client.post("/api/v1/models/predict?threshold=0.15", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert "risk_tier" in data
    assert "recommended_interventions" in data
    assert 0.0 <= data["readmission_probability"] <= 1.0


def test_report_generation_endpoint():
    payload = {
        "report_type": "clinical_audit",
        "decision_threshold": 0.15,
        "planned_encounters_count": 1000,
        "nurse_intervention_cost": 75.0,
        "avg_readmission_cost": 14500.0,
    }
    response = client.post("/api/v1/reports/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "markdown_content" in data
    assert "operations_impact" in data
