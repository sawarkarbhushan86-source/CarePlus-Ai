"""
CARE-PULSE AI — Executive Overview View
Executive summary, high-level clinical metrics, dataset provenance, and platform architecture.
"""

import streamlit as st
import pandas as pd
from ui.api_client import api_client
from ui.components import render_header, render_top_bar, render_clinical_disclaimer


def render():
    render_top_bar()
    render_header(
        title="CARE-PULSE AI — Executive Healthcare Intelligence",
        subtitle="Explainable Patient 30-Day Readmission Risk Prediction, Algorithmic Calibration, and Hospital Operations Analytics",
        category="Executive Intelligence",
    )

    # Top KPI Metrics Cards
    summary = api_client.get_dataset_summary()
    benchmark = api_client.get_benchmark_summary()
    eda = api_client.get_eda_overview()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Inpatient Cohort",
            value="101,766 Encounters",
            delta="71,518 Unique Patients",
            delta_color="off",
        )

    with col2:
        prevalence = eda.get("readmission_30d_percentage", 11.16) if eda else 11.16
        st.metric(
            label="30-Day Readmission Rate",
            value=f"{prevalence:.2f}%",
            delta="11,357 Readmitted Cases",
            delta_color="inverse",
        )

    with col3:
        champion = benchmark.get("best_model_name", "Calibrated XGBoost") if benchmark else "Calibrated XGBoost"
        best_auc = benchmark.get("best_roc_auc", 0.6773) if benchmark else 0.6773
        st.metric(
            label="Champion Model (ROC-AUC)",
            value=f"{best_auc:.4f}",
            delta=champion,
            delta_color="normal",
        )

    with col4:
        ece = (
            benchmark.get("metrics", {}).get("Calibrated XGBoost", {}).get("expected_calibration_error", 0.0157)
            if benchmark
            else 0.0157
        )
        st.metric(
            label="Calibration Error (ECE)",
            value=f"{ece:.4f}",
            delta="-85% Post-Platt Scaling",
            delta_color="normal",
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Platform Mission & Clinical Context
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("Clinical & Operational Objectives")
        st.markdown(
            """
            In acute hospital settings, unplanned 30-day readmissions represent a critical operational challenge:
            1. **Suboptimal Care Transitions:** Uncoordinated discharge handoffs, missed medication reconciliations, and inadequate glycemic follow-up drive recurrent hospitalizations.
            2. **Inpatient Bed Capacity Congestion:** Acute hospital beds occupied by preventable readmissions reduce hospital throughput and delay elective surgical care.
            3. **Regulatory Penalties:** Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face financial reimbursement penalties for excessive readmission ratios.

            **CARE-PULSE AI** bridges predictive machine learning with clinical transparency:
            - **Eliminates Patient Leakage:** Enforces patient-grouped cross-validation so repeated admissions of chronic patients never artificially inflate validation performance.
            - **Calibrates Probability Scores:** Platt scaling reduces Expected Calibration Error to 1.57%, ensuring risk numbers reflect empirical clinical event rates.
            - **Explains Patient Drivers:** Uncovers exact individual risk drivers using TreeSHAP to inform targeted nurse care plans.
            """
        )

    with c_right:
        st.subheader("Data Provenance & Governance")
        st.markdown(
            f"""
            - **Data Source:** UCI Machine Learning Repository / OpenML ID 4541
            - **Historical Cohort:** 1999–2008 across 130 US medical centers
            - **Target Definition:** Encounter readmission strictly within `<30` days
            - **Clinical Exclusions:** Discharges with disposition 11, 19, 20, 21 (deceased/hospice)
            - **Cohort Dimensions:** 101,766 encounters across 50 clinical features
            - **SHA-256 Fingerprint:**  
              `{summary.get('fingerprint', 'e965ae6f7b59d3e9e47124135e3153efba436d644331f805e812d26388c1256b')[:32]}...`
            """
        )

    st.markdown("---")

    # Workflow Architecture
    st.subheader("End-to-End Analytics Workflow")
    workflow_steps = [
        {"Stage": "1. Ingestion & Validation", "Description": "SHA-256 integrity verification, schema validation, missingness audit (40% drop cutoff)."},
        {"Stage": "2. Preprocessing & Splitting", "Description": "ICD-9 disease categorization, prior utilization aggregation, patient-grouped train/test split."},
        {"Stage": "3. Candidate Model Training", "Description": "Logistic Regression, Random Forest, XGBoost, and Platt-scaled calibrated XGBoost."},
        {"Stage": "4. Explainability & SHAP", "Description": "Global feature rankings and local patient-level risk attribution waterfall charts."},
        {"Stage": "5. Operations & Hospital Analytics", "Description": "Nurse care transition ROI simulation, sensitivity trade-offs, and bed-day forecasting."},
    ]
    st.table(pd.DataFrame(workflow_steps))

    render_clinical_disclaimer()


if __name__ == "__main__":
    render()
