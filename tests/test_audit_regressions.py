"""
CARE-PULSE AI — Comprehensive Senior Engineering & Scientific Validation Regression Suite
Validates target mapping, zero-leakage patient-grouped splitting, calibration consistency,
full dataset count reconciliation, ROI mathematical formulas, and safe report rendering.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.config import settings
from app.services.data_loader import data_loader_service
from app.services.data_validation import data_validation_service
from app.services.preprocessing import preprocessing_service
from app.services.operations_service import operations_service
from app.services.reporting_service import reporting_service
from app.schemas.report import ReportGenerationRequest


def test_scientific_binary_target_definition():
    """Verify strictly: <30 maps to 1, while >30 and NO map to 0."""
    raw_labels = pd.Series(["<30", ">30", "NO", "<30", "NO", ">30"])
    mapped = (raw_labels == "<30").astype(int)

    assert mapped.tolist() == [1, 0, 0, 1, 0, 0]
    assert mapped.sum() == 2
    # Verify no third class leakage
    assert set(mapped.unique()).issubset({0, 1})


def test_patient_grouped_splitting_zero_leakage():
    """Prove that patient identifiers have strictly 0% overlap between train and test splits."""
    df_sample = data_loader_service.load_dataset(preferred_source="sample", nrows=1000)
    split_res = preprocessing_service.prepare_modeling_data(df_sample, test_size=0.20, random_state=42)

    train_patients = set(split_res["raw_train_df"]["patient_nbr"])
    test_patients = set(split_res["raw_test_df"]["patient_nbr"])

    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0, f"Patient leakage detected! Overlapping patient IDs: {overlap}"


def test_clinical_exclusion_criteria():
    """Verify that deceased or hospice-discharged encounters are strictly flagged for exclusion."""
    # Discharge disposition IDs: 11 = Expired, 19 = Expired at home, 20 = Expired in medical facility, 21 = Expired place unknown
    mock_df = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [101, 102, 103, 104],
        "discharge_disposition_id": [1, 11, 3, 19],
        "gender": ["Female", "Male", "Female", "Male"],
    })
    exclusions = data_validation_service.identify_clinical_exclusions(mock_df)
    assert exclusions.iloc[1] is True or exclusions.iloc[1] == 1
    assert exclusions.iloc[3] is True or exclusions.iloc[3] == 1
    assert int(exclusions.sum()) == 2


def test_reconciled_full_cohort_metrics():
    """Verify full data warehouse reconciliation metrics match ground truth artifact."""
    val_file = settings.METRICS_DIR / "full_validation_report.json"
    assert val_file.exists(), "full_validation_report.json must be persisted"

    with open(val_file, "r") as f:
        data = json.load(f)

    assert data["total_rows"] == 101766
    assert data["unique_patients_count"] == 71518
    assert data["repeated_patient_encounters_count"] == 30248
    assert data["excluded_encounters_count"] == 1655
    assert 0.110 <= data["target_rate_30d"] <= 0.113
    assert data["target_distribution"]["<30"] == 11357
    assert data["target_distribution"][">30"] == 35545
    assert data["target_distribution"]["NO"] == 54864


def test_calibration_and_fairness_artifact_schema():
    """Verify reliability_summary.json exists, conforms to schema, and demonstrates calibration improvement."""
    summary_path = settings.METRICS_DIR / "reliability_summary.json"
    assert summary_path.exists(), "reliability_summary.json must exist"

    with open(summary_path, "r") as f:
        rel = json.load(f)

    # Required metadata keys
    assert "model_name" in rel
    assert "test_samples" in rel
    assert rel["test_samples"] == 4890
    assert "uncalibrated_ece" in rel
    assert "calibrated_ece" in rel
    assert "ece_reduction_pct" in rel

    # Calibrated ECE must be lower than raw ECE
    assert rel["calibrated_ece"] < rel["uncalibrated_ece"]
    assert rel["calibrated_ece"] < 0.05  # Standard clinical calibration bar

    # Calibration curve data coordinates must be valid
    assert "reliability_curve_cal" in rel
    assert len(rel["reliability_curve_cal"]["mean_predicted_value"]) > 0
    assert len(rel["reliability_curve_cal"]["fraction_of_positives"]) > 0

    # Demographic subgroup audit must be populated
    assert "subgroup_audit" in rel
    assert "race" in rel["subgroup_audit"]
    assert "gender" in rel["subgroup_audit"]
    assert "age" in rel["subgroup_audit"]
    assert "disparate_impact_ratio" in rel


def test_operations_roi_formulas_and_edge_cases():
    """Verify operations service formulas and edge cases (zero costs, zero cohort, sensitivity)."""
    # Standard baseline
    base = operations_service.simulate_intervention_program(
        cohort_size=2000,
        decision_threshold=0.15,
        avg_readmission_cost=14500.0,
        intervention_cost_per_patient=75.0,
        relative_risk_reduction=0.20,
        intervention_uptake_rate=0.80,
    )
    assert base.cohort_size == 2000
    assert base.predicted_high_risk > 0
    assert base.prevented_readmissions_estimate > 0
    # Gross savings formula
    expected_gross = base.prevented_readmissions_estimate * 14500.0
    assert base.gross_readmission_savings_usd == expected_gross
    # Program cost formula
    enrolled = int(round(base.predicted_high_risk * 0.80))
    expected_cost = enrolled * 75.0
    assert base.total_intervention_cost_usd == expected_cost
    # Net savings formula
    assert base.net_savings_usd == expected_gross - expected_cost
    # ROI formula
    expected_roi = ((expected_gross - expected_cost) / expected_cost) * 100.0
    assert abs(base.estimated_roi_percentage - round(expected_roi, 1)) < 0.2
    # Bed-days formula
    expected_beds = int(round(base.prevented_readmissions_estimate * 4.68))
    assert base.beds_freed_estimate == expected_beds

    # Edge case 1: Zero cohort size
    zero_cohort = operations_service.simulate_intervention_program(cohort_size=0)
    assert zero_cohort.cohort_size == 0
    assert zero_cohort.total_intervention_cost_usd == 0.0
    assert zero_cohort.estimated_roi_percentage == 0.0

    # Edge case 2: Zero intervention cost (costless intervention)
    zero_cost = operations_service.simulate_intervention_program(
        cohort_size=1000, intervention_cost_per_patient=0.0
    )
    assert zero_cost.total_intervention_cost_usd == 0.0
    assert zero_cost.estimated_roi_percentage >= 0.0  # Must not raise ZeroDivisionError

    # Edge case 3: Zero Relative Risk Reduction (ineffective intervention)
    zero_rrr = operations_service.simulate_intervention_program(
        cohort_size=1000, relative_risk_reduction=0.0
    )
    assert zero_rrr.prevented_readmissions_estimate == 0
    assert zero_rrr.gross_readmission_savings_usd == 0.0
    assert zero_rrr.net_savings_usd == -zero_rrr.total_intervention_cost_usd
    assert zero_rrr.estimated_roi_percentage == -100.0


def test_reporting_service_no_leaked_html_and_valid_evidence_framing():
    """Verify generated reports have no raw </div> leaks and use research-grade framing."""
    req = ReportGenerationRequest(report_type="clinical_audit", planned_encounters_count=1500)
    rep = reporting_service.generate_report(req)

    # Check for raw leaked HTML tags in markdown
    assert "</div>" not in rep.markdown_content, "Found leaked raw </div> tag in generated report markdown!"
    assert "<div>" not in rep.markdown_content, "Found leaked raw <div> tag in generated report markdown!"

    # Check for evidence-qualified research framing
    assert "Analytical Observations & Implementation Considerations" in rep.markdown_content
    assert "autonomous clinical directive" in rep.markdown_content.lower() or "disclaimer" in rep.markdown_content.lower()

    # Check that executive summary and operations impact are populated
    assert len(rep.executive_summary) > 50
    assert rep.operations_impact.cohort_size == 1500


def test_chi_square_repair_and_fisher_fallback():
    """Verify chi-square handles missing engineered columns, Cochran violations, and sparse Fisher fallback."""
    from app.services.statistics_service import statistics_service

    # Test auto-mapping diag_1_group from raw diag_1
    df_raw = pd.DataFrame({
        "diag_1": ["250.00", "428.0", "414.0", "250.02", "486", "780"],
        "readmission_30d": [1, 0, 0, 1, 0, 1],
    })
    res_diag = statistics_service.compute_chi_square_test(df_raw, "diag_1_group")
    assert res_diag["success"] is True
    assert "chi2_statistic" in res_diag
    assert "p_value" in res_diag

    # Test sparse 2x2 table triggering Fisher's exact test fallback
    df_sparse = pd.DataFrame({
        "rare_feature": ["A", "A", "A", "A", "B", "B"],
        "readmission_30d": [0, 0, 0, 0, 1, 0],
    })
    res_sparse = statistics_service.compute_chi_square_test(df_sparse, "rare_feature")
    assert res_sparse["success"] is True
    assert "Fisher's Exact Test" in res_sparse["test_used"]
    assert res_sparse["assumptions_met"] is False
    assert "Cochran" in res_sparse.get("cochran_warning", "")

    # Test non-existent column returns graceful error
    res_err = statistics_service.compute_chi_square_test(df_raw, "non_existent_col")
    assert res_err["success"] is False
    assert res_err["chi2_statistic"] == 0.0


def test_threshold_grid_evaluation():
    """Verify evaluation_service.evaluate_threshold_grid computes consistent trade-off metrics."""
    from app.services.evaluation_service import evaluation_service

    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 3 positives, 7 negatives (30% prevalence)
    y_prob = np.array([0.9, 0.7, 0.4, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])

    grid = evaluation_service.evaluate_threshold_grid(y_true, y_prob, thresholds=[0.10, 0.35, 0.80])
    assert len(grid) == 3

    # Low threshold (0.10)
    low_t = grid[0]
    assert low_t["threshold"] == 0.10
    assert low_t["flagged_patients"] >= grid[1]["flagged_patients"]
    assert low_t["recall_sensitivity"] >= grid[1]["recall_sensitivity"]
    assert low_t["true_positives"] + low_t["false_negatives"] == 3
    assert low_t["true_negatives"] + low_t["false_positives"] == 7

    # High threshold (0.80)
    high_t = grid[2]
    assert high_t["threshold"] == 0.80
    assert high_t["specificity"] >= low_t["specificity"]
    assert high_t["missed_readmission_count"] > 0
    assert high_t["workload_per_1000_discharges"] == int((high_t["flagged_patients"] / 10) * 1000)


def test_subgroup_fairness_wilson_ci_and_small_sample_warnings():
    """Verify subgroup analysis computes Wilson CIs and accurately flags under-powered cohorts."""
    from app.services.subgroup_analysis import subgroup_analysis_service, compute_wilson_score_interval

    # Wilson interval check
    low, high = compute_wilson_score_interval(3, 14)
    assert 0.05 < low < 0.15
    assert 0.35 < high < 0.55

    df_test = pd.DataFrame({
        "race": ["Asian"] * 10 + ["Caucasian"] * 120,
        "readmission_30d": [1] * 2 + [0] * 8 + [1] * 25 + [0] * 95,
    })

    y_true = df_test["readmission_30d"].values
    y_prob = np.random.RandomState(42).uniform(0.05, 0.30, len(df_test))

    audit = subgroup_analysis_service.evaluate_subgroups(df_test, y_true, y_prob, "race", threshold=0.15)
    asian_row = next(r for r in audit if r["subgroup"] == "Asian")
    cauc_row = next(r for r in audit if r["subgroup"] == "Caucasian")

    assert asian_row["sample_size"] == 10
    assert asian_row["positive_events_count"] == 2
    assert asian_row["small_sample_warning"] is True
    assert "Under-powered" in asian_row["sample_reliability_note"]

    assert cauc_row["sample_size"] == 120
    assert cauc_row["small_sample_warning"] is False
    assert "Adequate" in cauc_row["sample_reliability_note"]


def test_break_even_effectiveness_formula():
    """Verify break-even RRR formula and arithmetic."""
    be = operations_service.calculate_break_even_effectiveness(
        decision_threshold=0.15,
        intervention_cost_per_patient=75.0,
        avg_readmission_cost=14500.0,
        cohort_size=1000,
    )
    assert be["is_viable"] is True
    assert 0.01 <= be["break_even_rrr_percentage"] <= 10.0
    assert be["break_even_prevented_readmissions"] > 0
    assert be["total_program_cost"] > 0


def test_peak_bed_demand_modeling():
    """Verify average vs peak bed demand calculations."""
    cap = operations_service.calculate_bed_capacity_forecast(
        monthly_discharges=1000,
        baseline_readmission_rate=0.10,
        avg_readmission_los_days=4.5,
        total_licensed_beds=300,
        peak_surge_factor=1.20,
    )
    assert cap["average_daily_beds_occupied"] == 15.0  # (100 * 4.5) / 30 = 15.0
    assert cap["peak_bed_demand"] == 18.0  # 15.0 * 1.20 = 18.0
    assert cap["average_bed_capacity_share_pct"] == 5.0  # 15 / 300 = 5%
    assert cap["peak_bed_capacity_share_pct"] == 6.0  # 18 / 300 = 6%


def test_patient_simulator_non_prescriptive_and_category_fallback():
    """Verify patient simulator outputs are strictly non-prescriptive and handle invalid category fallbacks."""
    from app.services.prediction_service import prediction_service
    from app.schemas.prediction import PatientEncounterInput

    # Test invalid category fallback
    enc_invalid = PatientEncounterInput(
        race="Martian",  # Invalid race
        gender="UnknownBiologicalSex",  # Invalid gender
        age="999_years_old",  # Invalid age bracket
        insulin="ExtraMegaHigh",  # Invalid insulin status
        A1Cresult="UltraHigh12",  # Invalid HbA1c
    )
    res = prediction_service.predict_encounter(enc_invalid)
    assert res.is_simulated_record is True
    assert 0.0 <= res.readmission_probability <= 1.0

    # Test that recommendations are strictly non-prescriptive
    for rec in res.recommended_interventions:
        assert "Order Inpatient HbA1c Lab Test" not in rec
        assert "Standard discharge instructions" not in rec
        assert "Schedule 48-Hour" not in rec
    # Must include mandatory clinical review notice
    assert any("MANDATORY CLINICAL REVIEW" in rec for rec in res.recommended_interventions)

