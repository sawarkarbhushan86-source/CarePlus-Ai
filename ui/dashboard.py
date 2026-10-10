"""
CARE-PULSE AI — Streamlit Main Application Entrypoint
Explainable Healthcare Intelligence, Patient Readmission Risk Prediction,
and Hospital Analytics Platform.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.api_client import api_client
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

# 1. Configure Streamlit Page
st.set_page_config(
    page_title="CARE-PULSE AI | Healthcare Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Professional Healthcare Light Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean light background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #0284c7 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Navigation link hover and active states */
    a[data-testid="stSidebarNavLink"] {
        border-radius: 8px;
        padding: 8px 12px;
        margin: 2px 0;
        font-weight: 500;
        color: #334155;
        transition: all 0.15s ease-in-out;
    }
    a[data-testid="stSidebarNavLink"]:hover {
        background-color: #f1f5f9;
        color: #0284c7;
    }
    a[data-testid="stSidebarNavLink"][aria-current="page"] {
        background-color: #e0f2fe !important;
        color: #0284c7 !important;
        font-weight: 600 !important;
    }
    
    /* Form inputs and buttons */
    div.stButton > button[kind="primary"] {
        background-color: #0284c7;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369a1;
    }
    
    /* Disclaimers */
    .sidebar-footer {
        font-size: 11px;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 14px;
        margin-top: 24px;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar_status():
    """Render system status badges in sidebar."""
    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid #e2e8f0;">
                <div style="background: #e0f2fe; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 24px;">
                    🩺
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 17px; color: #0f172a; font-weight: 700; letter-spacing: -0.3px;">CARE-PULSE AI</h2>
                    <span style="font-size: 11px; color: #0284c7; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Healthcare Intelligence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar_footer():
    """Render footer in sidebar."""
    with st.sidebar:
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        health = api_client.get_health()
        api_alive = health.get("backend_connected", False)

        st.markdown(
            f"""
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 600;">Backend API:</span>
                    <span style="color: {'#10b981' if api_alive else '#0284c7'}; font-weight: 600;">
                        ● {'FastAPI Active' if api_alive else 'In-Process Direct'}
                    </span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 600;">Champion:</span>
                    <span style="color: #0f172a; font-weight: 600;">Calibrated XGB</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 600;">Inpatient Cohort:</span>
                    <span style="color: #0f172a; font-weight: 600;">101.7k Records</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #64748b; font-weight: 600;">Validation:</span>
                    <span style="color: #0284c7; font-weight: 600;">Patient-Grouped</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                <strong>CARE-PULSE AI Platform v1.0.0</strong><br>
                Historical cohort 1999–2008.<br>
                For research & portfolio evaluation only. Not for autonomous clinical decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# 3. Define 8-Page Navigation Structure
pages = [
    st.Page(overview.render, title="Overview", icon="🩺", url_path="overview", default=True),
    st.Page(data_explorer.render, title="Data Explorer", icon="📊", url_path="data-explorer"),
    st.Page(statistical_analysis.render, title="Statistical Analysis", icon="📈", url_path="statistical-analysis"),
    st.Page(model_lab.render, title="Model Lab", icon="🧪", url_path="model-lab"),
    st.Page(explainability.render, title="Explainability", icon="🔍", url_path="explainability"),
    st.Page(reliability.render, title="Reliability & Fairness", icon="⚖️", url_path="reliability-fairness"),
    st.Page(hospital_analytics.render, title="Hospital Analytics", icon="🏥", url_path="hospital-analytics"),
    st.Page(reports.render, title="Reports", icon="📋", url_path="reports"),
]

render_sidebar_status()
nav = st.navigation(pages)
nav.run()
render_sidebar_footer()
