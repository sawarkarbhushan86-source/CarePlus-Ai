"""
CARE-PULSE AI — Clinical and Executive Reports View
Generates exportable clinical audit reports, executive summaries, and operational care plans.
"""

from typing import Dict, Any, Optional
import streamlit as st
from ui.api_client import api_client
from app.schemas.report import ReportGenerationRequest
from ui.components import render_header, render_top_bar, render_clinical_disclaimer


def render():
    render_top_bar()
    render_header(
        title="Clinical Audit & Executive Reports",
        subtitle="Generate formal algorithmic audit reports, operational ROI analyses, and governance documentation for hospital leadership, data science committees, and technical evaluators.",
        category="Governance & Reporting",
    )

    with st.expander("Report Configuration Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            report_type = st.selectbox(
                "Select Report Type:",
                ["clinical_audit", "executive_summary", "operations_plan"],
                format_func=lambda x: {
                    "clinical_audit": "Algorithmic Audit & Model Card",
                    "executive_summary": "Executive Leadership Summary",
                    "operations_plan": "Operational Rollout & Care Transition Plan",
                }.get(x, x),
                index=0,
                key="report_type_select",
            )
            cohort_size = st.number_input(
                "Inpatient Annual Cohort Size:",
                min_value=100,
                max_value=500000,
                value=2000,
                step=250,
                key="cohort_size_input",
                help="Projected number of adult diabetic inpatient encounters admitted per year.",
            )

        with c2:
            threshold = st.slider(
                "Decision Threshold:",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
                key="threshold_slider",
                help="Mathematical decision cutoff for flagging patients for transitional care navigation.",
            )
            nurse_cost = st.number_input(
                "Nurse Transition Cost ($):",
                min_value=0.0,
                max_value=2000.0,
                value=75.0,
                step=5.0,
                key="nurse_cost_input",
                help="Direct cost per patient enrolled in post-discharge transitional care coordination.",
            )

        with c3:
            readm_cost = st.number_input(
                "Average Readmission Cost ($):",
                min_value=500.0,
                max_value=100000.0,
                value=14500.0,
                step=500.0,
                key="readm_cost_input",
                help="Average fully-loaded hospital inpatient readmission cost incurred within 30 days.",
            )
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button(
                "Generate Formal Report",
                type="primary",
                use_container_width=True,
                key="btn_generate_report",
            )

    # 1. Handle Click Action & Validation
    if generate_clicked:
        validation_errors = []
        if cohort_size is None or cohort_size < 100:
            validation_errors.append("Inpatient Annual Cohort Size must be at least 100 encounters.")
        if threshold is None or not (0.01 <= threshold <= 0.99):
            validation_errors.append("Decision threshold must be between 0.01 and 0.99.")
        if nurse_cost is None or nurse_cost < 0:
            validation_errors.append("Nurse transition cost must be non-negative ($0 or greater).")
        if readm_cost is None or readm_cost <= 0:
            validation_errors.append("Average readmission cost must be greater than $0.")

        if validation_errors:
            st.session_state["generated_report"] = None
            st.session_state["report_error"] = " | ".join(validation_errors)
        else:
            with st.spinner("Connecting to FastAPI backend and compiling formal report from active model artifacts..."):
                try:
                    req = ReportGenerationRequest(
                        report_type=report_type,
                        decision_threshold=float(threshold),
                        planned_encounters_count=int(cohort_size),
                        nurse_intervention_cost=float(nurse_cost),
                        avg_readmission_cost=float(readm_cost),
                    )
                    rep_res = api_client.generate_report(req, require_backend=True)
                    st.session_state["generated_report"] = rep_res
                    st.session_state["report_error"] = None
                    st.toast(f"Report {rep_res.get('report_id')} compiled successfully!", icon="📋")
                except Exception as exc:
                    st.session_state["generated_report"] = None
                    st.session_state["report_error"] = str(exc)

    # 2. Display Feedback & Status
    if st.session_state.get("report_error"):
        st.error(f"❌ **Report Generation Failed:** {st.session_state['report_error']}")

    current_report: Optional[Dict[str, Any]] = st.session_state.get("generated_report")
    has_valid_report = current_report is not None and bool(current_report.get("markdown_content"))

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # 3. Display Report Actions & Preview Header
    col_l, col_r = st.columns([3, 1])
    with col_l:
        if has_valid_report:
            st.subheader(f"Report Reference: `{current_report.get('report_id', 'REP-UNKNOWN')}`")
            st.caption(
                f"Generated at: {current_report.get('generated_at', 'N/A')} UTC | "
                f"Source: {current_report.get('source', 'FastAPI Backend')} | Engine Version: 1.0.0"
            )
        else:
            st.subheader("Report Document")
            st.caption("No report currently active. Configure parameters above and click 'Generate Formal Report'.")

    with col_r:
        if has_valid_report:
            st.download_button(
                label="📥 Download Markdown Report",
                data=current_report.get("markdown_content", ""),
                file_name=f"{current_report.get('report_id', 'report')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="btn_download_markdown",
            )
        else:
            st.button(
                label="📥 Download Markdown Report",
                disabled=True,
                help="Generate a formal report first to enable download.",
                use_container_width=True,
                key="btn_download_disabled",
            )

    # 4. Render Report Content or Empty State
    with st.container(border=True):
        if has_valid_report:
            st.markdown(current_report.get("markdown_content", ""))
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 48px 24px; color: #64748b; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 10px;">
                    <div style="font-size: 36px; margin-bottom: 12px;">📋</div>
                    <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 18px; font-weight: 600;">No Report Generated Yet</h3>
                    <p style="margin: 0; max-width: 520px; margin: 0 auto; font-size: 14px; line-height: 1.5; color: #64748b;">
                        Select your desired report type, annual cohort size, decision threshold, and cost parameters above,
                        then click <strong>Generate Formal Report</strong> to compile the formal algorithmic governance audit,
                        executive summary, or operational care transition rollout plan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_clinical_disclaimer()


if __name__ == "__main__":
    render()
