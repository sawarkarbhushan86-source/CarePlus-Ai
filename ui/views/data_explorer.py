"""
CARE-PULSE AI — Data Explorer View
Clinical encounter dataset inspection, data quality audit, missingness analysis,
schema profile, and interactive encounter preview.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui.api_client import api_client
from app.services.data_loader import data_loader_service
from ui.components import render_header, render_top_bar, render_clinical_disclaimer


def render():
    render_top_bar()
    render_header(
        title="Data Explorer & Quality Audit",
        subtitle="Encounter data hygiene, missingness rates, repeated patient revisit structures, and demographic cohort distributions.",
        category="Data Quality & Exploration",
    )

    # Validation Audit Summary
    val_report = api_client.get_validation_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Encounters", f"{val_report.get('total_rows', 101766):,}")
    with col2:
        st.metric("Unique Patients", f"{val_report.get('unique_patients_count', 71518):,}")
    with col3:
        st.metric("Repeated Encounters", f"{val_report.get('repeated_patient_encounters_count', 30248):,}")
    with col4:
        st.metric("30-Day Readmission Rate", f"{val_report.get('target_rate_30d', 0.1116) * 100:.2f}%")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution",
        "Missingness Audit",
        "Interactive Encounter Preview",
        "Schema & Feature Metadata",
    ])

    with tab1:
        st.subheader("Target Variable Distribution (Readmission Status)")
        raw_dist = val_report.get("target_distribution", {"NO": 54864, ">30": 35545, "<30": 11357})
        if raw_dist:
            labels = list(raw_dist.keys())
            values = list(raw_dist.values())

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.45,
                        marker=dict(colors=["#94a3b8", "#0284c7", "#ef4444"]),
                        textinfo="label+percent+value",
                    )
                ]
            )
            fig_pie.update_layout(
                title="Historical Readmission Distribution (UCI 130-US Hospitals)",
                margin=dict(t=40, b=20, l=20, r=20),
                height=380,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig_pie, theme="streamlit")

            st.markdown(
                """
                > **Clinical Formulation Note:**  
                > The primary positive classification target for CARE-PULSE AI is strictly `<30` (unplanned readmission within 30 days of discharge).  
                > Encounters labeled `>30` (readmitted after 30 days) and `NO` (no readmission recorded) are grouped into the negative class (0),  
                > aligning with the standard Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP) 30-day evaluation interval.
                """
            )

    with tab2:
        st.subheader("Feature Missingness Breakdown & Imputation Strategy")
        missing_rates = val_report.get("missingness_summary", {})
        if missing_rates:
            df_miss = (
                pd.DataFrame(list(missing_rates.items()), columns=["Feature", "Missing_Rate"])
                .sort_values(by="Missing_Rate", ascending=False)
            )
            df_miss["Missing_Percentage"] = df_miss["Missing_Rate"] * 100

            fig_bar = px.bar(
                df_miss.head(15),
                x="Feature",
                y="Missing_Percentage",
                color="Missing_Percentage",
                color_continuous_scale="Blues",
                title="Top Features by Missingness Percentage (%)",
                labels={"Missing_Percentage": "Missing (%)"},
            )
            fig_bar.add_hline(y=40.0, line_dash="dash", line_color="#ef4444", annotation_text="40% Pruning Threshold")
            fig_bar.update_layout(
                height=420,
                xaxis_tickangle=-45,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig_bar, theme="streamlit")

            st.markdown(
                """
                **Automated Cleaning & Hygiene Protocol:**  
                - `weight` (96.8% missing): Dropped from candidate modeling features due to extensive sparsity.
                - `payer_code` (39.5% missing) & `medical_specialty` (49.1% missing): Pruned or placed in dedicated unknown categories to prevent noisy estimations.
                - `race` (2.2% missing): Imputed with explicit `'Missing/Unknown'` categorical bucket to preserve encounter representation without patient dropping.
                """
            )

    with tab3:
        st.subheader("Interactive Encounter Explorer & Filter")
        st.markdown(
            "Inspect encounter records with customizable column selection and filtering. "
            "**Data Scope Note:** To ensure smooth in-browser rendering, an interactive exploration sample is loaded below, "
            "while population metrics above reflect the full **101,766 encounter** clinical data warehouse."
        )

        try:
            sample_size = st.slider("Exploration Sample Size Loaded (Rows):", min_value=100, max_value=2000, value=500, step=100)
            df_sample = data_loader_service.load_dataset(preferred_source="sample", nrows=sample_size)
            
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                gender_filter = st.multiselect(
                    "Filter by Gender:",
                    options=list(df_sample["gender"].dropna().unique()),
                    default=list(df_sample["gender"].dropna().unique())[:2],
                )
            with f_col2:
                age_filter = st.multiselect(
                    "Filter by Age Decile:",
                    options=sorted(list(df_sample["age"].dropna().unique())),
                    default=sorted(list(df_sample["age"].dropna().unique()))[4:8],
                )
            with f_col3:
                readm_filter = st.multiselect(
                    "Filter by Readmission Outcome:",
                    options=list(df_sample["readmitted"].dropna().unique()),
                    default=list(df_sample["readmitted"].dropna().unique()),
                )

            # Apply filters
            df_filtered = df_sample[
                (df_sample["gender"].isin(gender_filter))
                & (df_sample["age"].isin(age_filter))
                & (df_sample["readmitted"].isin(readm_filter))
            ]

            # Dynamic Scope Reconciliation Metrics
            f_m1, f_m2, f_m3, f_m4 = st.columns(4)
            n_filtered = len(df_filtered)
            n_pts_filtered = df_filtered["patient_nbr"].nunique() if n_filtered > 0 else 0
            rate_30d_filtered = (df_filtered["readmitted"] == "<30").mean() * 100.0 if n_filtered > 0 else 0.0
            
            f_m1.metric("Filtered Encounters", f"{n_filtered:,}", f"of {len(df_sample):,} sampled")
            f_m2.metric("Unique Patients (Subset)", f"{n_pts_filtered:,}")
            f_m3.metric("Subset 30d Readmit Rate", f"{rate_30d_filtered:.1f}%")
            cnt_lt30 = int((df_filtered["readmitted"] == "<30").sum()) if n_filtered > 0 else 0
            cnt_gt30 = int((df_filtered["readmitted"] == ">30").sum()) if n_filtered > 0 else 0
            cnt_no = int((df_filtered["readmitted"] == "NO").sum()) if n_filtered > 0 else 0
            f_m4.metric("Subset Class Breakdown", f"<30: {cnt_lt30} | >30: {cnt_gt30} | NO: {cnt_no}")

            default_columns = [
                "encounter_id",
                "patient_nbr",
                "race",
                "gender",
                "age",
                "time_in_hospital",
                "num_medications",
                "number_inpatient",
                "diag_1",
                "A1Cresult",
                "insulin",
                "readmitted",
            ]
            selected_cols = st.multiselect(
                "Columns to display:",
                options=list(df_sample.columns),
                default=[c for c in default_columns if c in df_sample.columns],
            )

            st.dataframe(df_filtered[selected_cols].head(200), height=320)
            st.caption(
                f"Displaying top {min(n_filtered, 200)} of {n_filtered} matching encounters from active sample "
                f"({len(df_sample)} sampled from {val_report.get('total_rows', 101766):,} full repository encounters)."
            )

            # CSV Download
            csv_data = df_filtered[selected_cols].to_csv(index=False)
            st.download_button(
                label=f"Download Filtered Subset ({n_filtered} Records, CSV)",
                data=csv_data,
                file_name=f"care_pulse_filtered_subset_n{n_filtered}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error loading sample encounter records: {e}")

    with tab4:
        st.subheader("Dataset Schema & Dimensionality")
        summary = api_client.get_dataset_summary()
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total Records", f"{summary.get('total_rows', 101766):,}")
        col_s2.metric("Total Features", f"{summary.get('total_columns', 50)}")
        col_s3.metric("Data Provenance", "UCI / OpenML 4541")

        st.markdown("**Core Feature Groups:**")
        feature_groups = [
            {"Group": "Identifiers", "Count": "2", "Examples": "encounter_id, patient_nbr"},
            {"Group": "Demographics", "Count": "3", "Examples": "race, gender, age"},
            {"Group": "Admission / Discharge", "Count": "4", "Examples": "admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital"},
            {"Group": "Clinical Utilization", "Count": "6", "Examples": "num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient"},
            {"Group": "Diagnosis Codes (ICD-9)", "Count": "4", "Examples": "diag_1, diag_2, diag_3, number_diagnoses"},
            {"Group": "Glycemic Testing", "Count": "2", "Examples": "max_glu_serum, A1Cresult"},
            {"Group": "Medications (23 agents)", "Count": "23", "Examples": "metformin, glipizide, glyburide, insulin, pioglitazone"},
            {"Group": "Management & Target", "Count": "3", "Examples": "change, diabetesMed, readmitted"},
        ]
        st.table(pd.DataFrame(feature_groups))

    render_clinical_disclaimer()


if __name__ == "__main__":
    render()
