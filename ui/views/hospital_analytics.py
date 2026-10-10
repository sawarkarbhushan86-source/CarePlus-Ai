"""
CARE-PULSE AI — Hospital Analytics & Operations Simulator View
Models bed capacity constraints, inpatient throughput, break-even effectiveness,
and Return on Investment (ROI) for targeted post-discharge nurse care transition programs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ui.api_client import api_client
from app.services.operations_service import operations_service
from ui.components import render_header, render_top_bar, render_clinical_disclaimer


def render():
    render_top_bar()
    render_header(
        title="Hospital Analytics & Operations Simulator",
        subtitle="Translating machine learning risk scores into hospital bed capacity planning, break-even thresholds, and financial Return on Investment (ROI) for targeted nurse transition programs.",
        category="Operational Health Systems",
    )

    tab1, tab2 = st.tabs([
        "Care Transition Program ROI & Break-Even",
        "Inpatient Bed Capacity Forecasting (Average & Peak)",
    ])

    with tab1:
        st.subheader("Targeted Nurse Care Transition Intervention ROI")
        st.markdown(
            "Simulate the financial and operational impact of enrolling patients identified by CARE-PULSE AI "
            "as High-Risk into a structured post-discharge outreach program (nurse phone navigation + medication reconciliation)."
        )

        st.warning(
            "**Hypothetical Operational Projection:** All metrics below are prospective mathematical simulations "
            "grounded in published clinical trials (e.g., Coleman et al. 2006, Naylor et al. 2004) and empirical test cohort precision. "
            "They represent modeled planning scenarios, not observed historical clinical outcomes."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            cohort_size = st.slider("Target Inpatient Cohort (Encounters/Year):", 500, 10000, 2000, step=250)
            threshold = st.slider("Decision Threshold (Risk Cutoff for Enrollment):", 0.05, 0.30, 0.15, step=0.01)

        with c2:
            uptake_pct = st.slider("Patient Enrollment / Uptake Rate (%):", 50, 100, 80, step=5, help="Accounts for patients who decline participation or cannot be contacted post-discharge.")
            rrr_pct = st.slider("Base Case Relative Risk Reduction (%):", 10, 35, 20, step=1, help="Expected proportional reduction in 30-day readmissions based on transitional care trials.")

        with c3:
            nurse_cost = st.slider("Nurse Intervention Cost per Enrolled Patient ($):", 25, 250, 75, step=5)
            readm_cost = st.slider("Average Cost per 30-Day Readmission ($):", 8000, 25000, 14500, step=500)

        # Run primary simulation
        ops = api_client.simulate_operations(
            cohort_size=cohort_size,
            decision_threshold=threshold,
            nurse_cost=nurse_cost,
            readmission_cost=readm_cost,
            uptake_rate=uptake_pct / 100.0,
            relative_risk_reduction=rrr_pct / 100.0,
        )

        # Break-Even Calculation
        be = operations_service.calculate_break_even_effectiveness(
            decision_threshold=threshold,
            intervention_cost_per_patient=nurse_cost,
            avg_readmission_cost=readm_cost,
            cohort_size=cohort_size,
            intervention_uptake_rate=uptake_pct / 100.0,
        )

        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

        # KPI Metrics Row 1
        m1, m2, m3, m4 = st.columns(4)
        enrolled_count = int(round(ops['predicted_high_risk'] * (uptake_pct / 100.0)))
        m1.metric("Enrolled High-Risk Patients", f"{enrolled_count:,}", f"of {ops['predicted_high_risk']:,} flagged")
        m2.metric("Prevented Readmissions", f"{ops['prevented_readmissions_estimate']:,}")
        m3.metric("Net Financial Savings", f"${ops['net_savings_usd']:,.2f}")
        m4.metric("Program ROI", f"{ops['estimated_roi_percentage']:.1f}%", delta="Positive Return" if ops['estimated_roi_percentage'] > 0 else "Net Loss", delta_color="normal")

        # Break-Even Analysis Banner
        st.markdown(
            f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; padding: 14px 18px; border-radius: 8px; margin: 16px 0; font-size: 13px; line-height: 1.6; color: #166534;">
                <strong>Break-Even Operational Analysis:</strong><br>
                • <strong>Break-Even RRR:</strong> The nurse outreach program achieves financial self-sufficiency at a <strong>Relative Risk Reduction of only {be['break_even_rrr_percentage']:.2f}%</strong>.<br>
                • <strong>Averted Events Required:</strong> Preventing just <strong>{be['break_even_prevented_readmissions']} readmissions</strong> across the entire cohort fully offsets the total program operating cost of <strong>${be['total_program_cost']:,.2f}</strong>.<br>
                • <strong>Underlying Precision:</strong> Model Positive Predictive Value at threshold {threshold:.2f} is {be['precision_at_threshold'] * 100:.1f}%.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mathematical Formula Transparency Card
        with st.expander("Audit Mathematical Formulas & Assumptions", expanded=False):
            st.markdown(
                f"""
                - **Flagged Volume:** ${ops['predicted_high_risk']:,}$ patients (empirically derived from test-set precision at threshold ${threshold:.2f}$).
                - **Enrolled Cohort:** ${ops['predicted_high_risk']:,} \\times {uptake_pct}\\% = {enrolled_count:,}$ patients.
                - **Expected Readmissions in Enrolled:** ${enrolled_count:,} \\times {be['precision_at_threshold'] * 100:.1f}\\%\\text{{ Precision}} = {enrolled_count * be['precision_at_threshold']:.1f}$ events.
                - **Prevented Readmissions:** ${enrolled_count * be['precision_at_threshold']:.1f} \\times {rrr_pct}\\%\\text{{ RRR}} = {ops['prevented_readmissions_estimate']:,}$ readmissions.
                - **Program Cost:** ${enrolled_count:,} \\text{{ enrolled}} \\times \\${nurse_cost} = \\${ops['total_intervention_cost_usd']:,.2f}$.
                - **Gross Savings:** ${ops['prevented_readmissions_estimate']:,} \\text{{ prevented}} \\times \\${readm_cost:,.2f} = \\${ops['gross_readmission_savings_usd']:,.2f}$.
                - **Net Savings:** $\\text{{Gross Savings}} - \\text{{Program Cost}} = \\${ops['net_savings_usd']:,.2f}$.
                - **ROI (%):** $(\\text{{Net Savings}} / \\text{{Program Cost}}) \\times 100 = {ops['estimated_roi_percentage']:.1f}\\%$ (safe against zero-cost denominators).
                - **Break-Even Formula:** $\\text{{RRR}}_{{\\text{{break-even}}}} = \\frac{{\\text{{Nurse Cost}}}}{{\\text{{Precision}} \\times \\text{{Readmission Cost}}}} = \\frac{{{nurse_cost}}}{{{be['precision_at_threshold']:.4f} \\times {readm_cost}}} = {be['break_even_rrr_percentage']:.2f}\\%$.
                - **Bed-Days Freed:** ${ops['prevented_readmissions_estimate']:,} \\times 4.68\\text{{ days (empirical ALOS)}} = {ops['beds_freed_estimate']:,}$ bed-days.
                """
            )

        # Visual financial waterfall
        st.subheader("Financial Impact Breakdown")
        fig_waterfall = go.Figure(
            go.Waterfall(
                name="Financials",
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["Gross Readmission Savings", "Intervention Program Costs", "Net Hospital Savings"],
                textposition="outside",
                text=[
                    f"+${ops['gross_readmission_savings_usd']:,.0f}",
                    f"-${ops['total_intervention_cost_usd']:,.0f}",
                    f"${ops['net_savings_usd']:,.0f}",
                ],
                y=[
                    ops["gross_readmission_savings_usd"],
                    -ops["total_intervention_cost_usd"],
                    ops["net_savings_usd"],
                ],
                connector={"line": {"color": "#94a3b8"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#0284c7"}},
            )
        )
        fig_waterfall.update_layout(
            height=380,
            margin=dict(t=40, b=30, l=40, r=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            yaxis=dict(gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig_waterfall, theme="streamlit")

        # Multi-Scenario Sensitivity Analysis with Editable Scenarios
        st.subheader("Multi-Scenario Sensitivity Analysis (Editable Scenarios)")
        st.markdown("Adjust effectiveness scenarios below to model conservative, base, and optimistic interventional outcomes:")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            rrr_cons = st.number_input("Conservative RRR (%):", min_value=5, max_value=25, value=12, step=1)
        with sc2:
            rrr_base = st.number_input("Base Case RRR (%):", min_value=10, max_value=35, value=int(rrr_pct), step=1)
        with sc3:
            rrr_opt = st.number_input("Optimistic RRR (%):", min_value=15, max_value=50, value=28, step=1)

        sens_cases = [
            {"Scenario": f"Conservative (RRR {rrr_cons}%)", "RRR": rrr_cons / 100.0},
            {"Scenario": f"Base Case (RRR {rrr_base}%)", "RRR": rrr_base / 100.0},
            {"Scenario": f"Optimistic (RRR {rrr_opt}%)", "RRR": rrr_opt / 100.0},
        ]
        sens_rows = []
        for c in sens_cases:
            s_ops = api_client.simulate_operations(
                cohort_size=cohort_size,
                decision_threshold=threshold,
                nurse_cost=nurse_cost,
                readmission_cost=readm_cost,
                uptake_rate=uptake_pct / 100.0,
                relative_risk_reduction=c["RRR"],
            )
            sens_rows.append({
                "Planning Scenario": c["Scenario"],
                "Prevented Readmissions": f"{s_ops['prevented_readmissions_estimate']:,}",
                "Program Cost": f"${s_ops['total_intervention_cost_usd']:,.0f}",
                "Gross Savings": f"${s_ops['gross_readmission_savings_usd']:,.0f}",
                "Net Savings": f"${s_ops['net_savings_usd']:,.0f}",
                "Projected ROI": f"{s_ops['estimated_roi_percentage']:.1f}%",
                "Bed-Days Freed": f"{s_ops['beds_freed_estimate']:,}",
            })
        st.table(pd.DataFrame(sens_rows))

        st.info(
            f"**Bed Capacity Relief:** The estimated {ops['prevented_readmissions_estimate']} prevented readmissions "
            f"liberate approximately **{ops['beds_freed_estimate']} acute inpatient bed-days** annually "
            f"(assuming the empirical 4.68-day average hospital stay for diabetic inpatients), "
            "reducing Emergency Department boarding delays and accommodating elective surgical admissions."
        )

    with tab2:
        st.subheader("Hospital Bed Occupancy Pressure Modeling (Average vs Peak)")
        st.markdown(
            "Forecast the baseline acute bed volume consumed by recurrent diabetic readmissions, "
            "distinguishing steady-state average occupancy from peak demand surges."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            discharges = st.number_input("Monthly Diabetic Inpatient Discharges:", value=1200, step=100)
            licensed_beds = st.number_input("Total Licensed Hospital Inpatient Beds:", value=350, step=25)

        with col_b:
            avg_stay = st.number_input("Average Inpatient Length of Stay (Days):", value=4.7, step=0.1)
            target_readm_rate = st.slider("Baseline Readmission Rate (%):", 5.0, 20.0, 11.2, step=0.1)

        surge_factor = st.slider(
            "Peak Occupancy Surge Multiplier:",
            min_value=1.10,
            max_value=1.40,
            value=1.20,
            step=0.05,
            help="Multiplier to capture midweek admission spikes and seasonal respiratory census peaks.",
        )

        cap_data = operations_service.calculate_bed_capacity_forecast(
            monthly_discharges=discharges,
            baseline_readmission_rate=target_readm_rate / 100.0,
            avg_readmission_los_days=avg_stay,
            total_licensed_beds=licensed_beds,
            peak_surge_factor=surge_factor,
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Annual Readmissions", f"{cap_data['projected_annual_readmissions']:,}")
        b2.metric("Annual Bed-Days Consumed", f"{cap_data['annual_readmission_bed_days']:,.0f}")
        b3.metric("Average Daily Beds", f"{cap_data['average_daily_beds_occupied']} beds", f"{cap_data['average_bed_capacity_share_pct']}% capacity")
        b4.metric("Peak Bed Demand", f"{cap_data['peak_bed_demand']} beds", f"{cap_data['peak_bed_capacity_share_pct']}% capacity")

        st.markdown(
            f"""
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 16px; font-size: 13px; color: #334155; line-height: 1.6;">
                <strong>Bed Capacity Planning Assumptions & Disclosure:</strong><br>
                • <strong>Average Continuous Occupancy:</strong> An average of <strong>{cap_data['average_daily_beds_occupied']} licensed beds</strong> are continuously occupied year-round by readmitted patients.<br>
                • <strong>Peak Demand Modeling:</strong> During peak demand spikes (modeled with a {surge_factor:.2f}x multiplier), readmission census reaches <strong>{cap_data['peak_bed_demand']} beds</strong> ({cap_data['peak_bed_capacity_share_pct']}% of hospital capacity).<br>
                • <strong>Disclosed Assumptions:</strong> Surge factor incorporates historical empirical patterns of Tuesday–Thursday admission census peaks, weekend discharge slowdowns, and winter respiratory illness surges.
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_clinical_disclaimer()


if __name__ == "__main__":
    render()
