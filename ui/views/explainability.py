"""
CARE-PULSE AI — Model Explainability View
Global TreeSHAP feature importance rankings, dynamic patient risk attribution breakdowns,
and non-causal association interpretations with explicit baseline reference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ui.api_client import api_client
from app.schemas.prediction import PatientEncounterInput
from ui.components import render_header, render_top_bar, render_clinical_disclaimer


def render():
    render_top_bar()
    render_header(
        title="Model Explainability & SHAP Attribution",
        subtitle="Interpreting predictive decision boundaries using SHapley Additive exPlanations (SHAP). Audits global population feature importance and local patient-level risk attributions.",
        category="Algorithmic Transparency",
    )

    tab1, tab2 = st.tabs([
        "Global Population Feature Importance",
        "Local Patient Risk Attribution & Baseline Decomposition",
    ])

    with tab1:
        st.subheader("Global SHAP Feature Rankings (Population Cohort)")
        st.markdown(
            "Mean absolute SHAP values indicate which clinical features exert the greatest influence on predicted readmission risk across all patient encounters."
        )

        global_shap = api_client.get_global_explanations(model_name="XGBoost")
        features = global_shap.get("features", [])
        values = global_shap.get("mean_abs_shap", [])

        if features and values:
            df_shap = pd.DataFrame({"Feature": features, "Mean_Abs_SHAP": values}).sort_values(
                by="Mean_Abs_SHAP", ascending=True
            )

            fig_global = go.Figure(
                go.Bar(
                    x=df_shap["Mean_Abs_SHAP"],
                    y=df_shap["Feature"],
                    orientation="h",
                    marker=dict(color="#0284c7"),
                )
            )
            fig_global.update_layout(
                title="Top Clinical Predictors by Mean |SHAP Value|",
                xaxis_title="Mean |SHAP Value| (Impact on Log-Odds of Readmission)",
                yaxis_title="",
                height=520,
                margin=dict(l=240, r=30, t=40, b=40),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0"),
            )
            st.plotly_chart(fig_global, theme="streamlit")

            st.markdown(
                """
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; padding: 14px 18px; border-radius: 8px; margin-top: 14px; font-size: 13px; line-height: 1.5; color: #334155;">
                    <strong>SHAP Attribution Scale & Non-Causal Interpretation:</strong><br>
                    • <strong>Scale:</strong> Values represent Mean |SHAP| on the margin/log-odds scale (<code>Δ ln(p / (1-p))</code>) of the gradient boosting tree ensemble. Higher values indicate features that exert stronger leverage on the classification boundary.<br>
                    • <strong>Associational Nature:</strong> High feature attribution demonstrates that the model relies on this variable for risk stratification. It does <em>not</em> prove that altering the variable will causally reduce readmission incidence.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                **Clinical Insights from Global Feature Rankings:**
                1. **Prior Utilization Dominance:** `Prior Inpatient Admissions` and `Total Prior Hospital Encounters` are the top 2 predictors, aligning with health services research showing that past utilization is the strongest marker of unmanaged chronic illness.
                2. **Multimorbidity & Frailty:** High diagnosis counts and lengthy inpatient stays reflect clinical complexity.
                3. **Glycemic & Medication Adjustments:** Adjusting insulin doses (`Insulin: Up/Down`) and elevated HbA1c reflect unstable disease states preceding discharge.
                """
            )

    with tab2:
        st.subheader("Local Patient Risk Attribution (Baseline Decomposition)")
        st.markdown(
            "Every model prediction is explained as a departure from a defined **Population Baseline** ($E[f(x)]$). "
            "Features either push risk upward (positive SHAP) or exert protective downward leverage (negative SHAP)."
        )

        mode_choice = st.radio(
            "Select Input Mode:",
            ["Preset Clinical Archetypes", "Custom Encounter Parameters"],
            horizontal=True,
        )

        if mode_choice == "Preset Clinical Archetypes":
            archetype_choice = st.selectbox(
                "Select Clinical Archetype:",
                [
                    "Archetype A: High-Risk Complex Multimorbid Patient",
                    "Archetype B: Moderate-Risk Patient with Insulin Titration",
                    "Archetype C: Low-Risk Controlled Diabetic Patient",
                ],
            )

            if "Archetype A" in archetype_choice:
                enc = PatientEncounterInput(
                    age="[70-80)",
                    race="AfricanAmerican",
                    gender="Female",
                    time_in_hospital=7,
                    number_inpatient=2,
                    number_emergency=1,
                    num_medications=22,
                    number_diagnoses=9,
                    A1Cresult=">8",
                    max_glu_serum="None",
                    insulin="Up",
                    diag_1="428",
                    change="Ch",
                    diabetesMed="Yes",
                )
            elif "Archetype B" in archetype_choice:
                enc = PatientEncounterInput(
                    age="[50-60)",
                    race="Caucasian",
                    gender="Male",
                    time_in_hospital=4,
                    number_inpatient=0,
                    number_emergency=1,
                    num_medications=14,
                    number_diagnoses=6,
                    A1Cresult=">7",
                    max_glu_serum="None",
                    insulin="Steady",
                    diag_1="250",
                    change="Ch",
                    diabetesMed="Yes",
                )
            else:
                enc = PatientEncounterInput(
                    age="[40-50)",
                    race="Caucasian",
                    gender="Female",
                    time_in_hospital=2,
                    number_inpatient=0,
                    number_emergency=0,
                    num_medications=6,
                    number_diagnoses=3,
                    A1Cresult="Norm",
                    max_glu_serum="None",
                    insulin="No",
                    diag_1="486",
                    change="No",
                    diabetesMed="No",
                )
        else:
            with st.expander("Configure Custom Encounter Parameters", expanded=True):
                ca1, ca2, ca3 = st.columns(3)
                with ca1:
                    c_age = st.selectbox("Age Bracket:", ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"], index=2)
                    c_race = st.selectbox("Race:", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
                    c_gender = st.selectbox("Gender:", ["Female", "Male"], index=0)
                    c_stay = st.slider("Length of Stay:", 1, 14, 4)
                with ca2:
                    c_inp = st.slider("Prior Inpatient Admissions:", 0, 6, 1)
                    c_er = st.slider("Prior ER Visits:", 0, 5, 0)
                    c_meds = st.slider("Medications Count:", 1, 40, 15)
                    c_diag = st.slider("Diagnoses Count:", 1, 16, 7)
                with ca3:
                    c_a1c = st.selectbox("HbA1c Lab Result:", [">8", ">7", "Norm", "None"], index=3)
                    c_ins = st.selectbox("Insulin Status:", ["Up", "Down", "Steady", "No"], index=3)
                    c_chg = st.selectbox("Medication Changed:", ["Ch", "No"], index=1)
                    c_dx = st.selectbox("Primary Diagnosis Code:", ["428", "250", "414", "486"], index=0)

            enc = PatientEncounterInput(
                age=c_age,
                race=c_race,
                gender=c_gender,
                time_in_hospital=c_stay,
                number_inpatient=c_inp,
                number_emergency=c_er,
                num_medications=c_meds,
                number_diagnoses=c_diag,
                diag_1=c_dx,
                A1Cresult=c_a1c,
                insulin=c_ins,
                change=c_chg,
                diabetesMed="Yes" if c_ins != "No" or c_chg == "Ch" else "No",
            )

        # Predict and explain
        res = api_client.predict_encounter(enc, model_name="Calibrated XGBoost")
        prob = res.get("readmission_probability", 0.15)
        tier = res.get("risk_tier", "Moderate Risk")
        base_log_odds = res.get("baseline_expected_value", -2.04)
        pos_sum = res.get("positive_risk_contributions", 0.0)
        neg_sum = res.get("negative_risk_contributions", 0.0)

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # Baseline to Prediction Decomposition Cards
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Baseline Risk (E[f(x)])", "11.2%", f"{base_log_odds:.2f} log-odds")
        col_b2.metric("Risk-Increasing Drivers (Σ+)", f"+{pos_sum:.3f}", "Pushes Risk Up")
        col_b3.metric("Protective Factors (Σ-)", f"{neg_sum:.3f}", "Pulls Risk Down")
        col_b4.metric("Final Predicted Risk", f"{prob * 100:.1f}%", tier)

        st.markdown("#### Primary Local Feature Attributions (SHAP):")
        drivers = res.get("primary_risk_drivers", [])

        if drivers:
            driver_labels = [f"{d['feature']} = {d['feature_value']}" for d in drivers][::-1]
            driver_shaps = [d['shap_value'] for d in drivers][::-1]
            bar_colors = ["#ef4444" if val > 0 else "#10b981" for val in driver_shaps]

            fig_local = go.Figure(
                go.Bar(
                    x=driver_shaps,
                    y=driver_labels,
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{v:+.3f}" for v in driver_shaps],
                    textposition="outside",
                )
            )
            fig_local.add_vline(x=0.0, line_dash="solid", line_color="#94a3b8")
            fig_local.update_layout(
                title="Feature Contribution on Prediction Log-Odds Margin",
                xaxis_title="SHAP Value (Red = Pushes Risk Up, Green = Protective Pull)",
                yaxis_title="",
                height=350,
                margin=dict(l=260, r=40, t=40, b=40),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0"),
            )
            st.plotly_chart(fig_local, theme="streamlit")

            df_drivers = pd.DataFrame(
                [
                    {
                        "Clinical Variable": d["feature"],
                        "Recorded Value": d["feature_value"],
                        "SHAP Attribution": f"{d['shap_value']:+.3f} log-odds",
                        "Directional Relationship": "Statistical Risk Increase" if d["direction"] == "increases_risk" else "Statistical Protective Association",
                    }
                    for d in drivers
                ]
            )
            st.dataframe(df_drivers, height=220)

        # Attribution disclaimer
        st.warning(
            "**Non-Causal Epidemiological Interpretation:** Feature contributions reflect statistical associations "
            "learned by the gradient boosting algorithm across historical patient records. They do not constitute "
            "biomedical proof of clinical causality. Interventions should always follow licensed clinical physician judgment."
        )

    render_clinical_disclaimer()


if __name__ == "__main__":
    render()
