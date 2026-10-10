"""
CARE-PULSE AI — Streamlit UI Components & Design System
Reusable clinical UI components adhering to modern healthcare design principles:
clean light theme, deep navy typography, restrained blue/teal accents, responsive cards.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import streamlit as st
from ui.api_client import api_client


def render_header(title: str, subtitle: str, category: Optional[str] = None):
    """Render a clean, modern clinical page header with typography hierarchy."""
    category_html = (
        f'<div style="color: #0284c7; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">{category}</div>'
        if category
        else ""
    )
    st.markdown(
        f"""
        <div style="background: #ffffff; padding: 22px 28px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04); margin-bottom: 24px;">
            {category_html}
            <h1 style="color: #0f172a; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">
                {title}
            </h1>
            <p style="color: #64748b; margin: 6px 0 0 0; font-size: 14px; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar():
    """Render the top status bar displaying backend status, dataset integrity, and last sync."""
    health = api_client.get_health()
    api_alive = health.get("backend_connected", False)
    sync_time = datetime.now().strftime("%H:%M:%S")

    col1, col2, col3, col4 = st.columns([2.5, 2.5, 2.5, 1.5])

    with col1:
        backend_color = "#10b981" if api_alive else "#0284c7"
        backend_label = "FastAPI Service Connected" if api_alive else "Direct In-Process Engine"
        st.markdown(
            f"""
            <div style="background: #ffffff; padding: 8px 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; display: flex; align-items: center; gap: 8px;">
                <span style="color: {backend_color}; font-size: 14px;">●</span>
                <span style="color: #475569; font-weight: 600;">Backend:</span>
                <span style="color: #0f172a; font-weight: 500;">{backend_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: #ffffff; padding: 8px 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #10b981; font-size: 14px;">✓</span>
                <span style="color: #475569; font-weight: 600;">Cohort:</span>
                <span style="color: #0f172a; font-weight: 500;">101,766 Inpatient Encounters</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background: #ffffff; padding: 8px 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #0284c7; font-size: 14px;">★</span>
                <span style="color: #475569; font-weight: 600;">Champion:</span>
                <span style="color: #0f172a; font-weight: 500;">{health.get('champion_model', 'Calibrated XGB')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="background: #ffffff; padding: 8px 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; text-align: center;">
                <span style="color: #64748b; font-size: 11px;">Sync: {sync_time}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)


def render_clinical_disclaimer():
    """Render explicit clinical and portfolio disclaimer."""
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 8px; margin-top: 32px; font-size: 12px; color: #64748b; line-height: 1.5;">
            <strong style="color: #0f172a;">Research & Clinical Governance Notice:</strong><br>
            CARE-PULSE AI is an explainable machine learning platform engineered for clinical data science portfolio demonstration, 
            academic health services research, and operational capacity planning. It evaluates historical inpatient encounters 
            (UCI 130-US Medical Centers cohort, 1999–2008). <em>This platform is not an FDA-cleared Software as a Medical Device (SaMD) 
            and must not be utilized for autonomous clinical diagnosis, treatment alteration, or discharge decisions.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str, action_hint: Optional[str] = None):
    """Render a clean, actionable empty state card."""
    action_html = f'<p style="color: #0284c7; font-size: 13px; font-weight: 600; margin-top: 8px;">👉 {action_hint}</p>' if action_hint else ""
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 36px 24px; text-align: center; margin: 24px 0;">
            <div style="font-size: 36px; margin-bottom: 12px;">📁</div>
            <h3 style="color: #0f172a; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">{title}</h3>
            <p style="color: #64748b; font-size: 14px; max-width: 500px; margin: 0 auto; line-height: 1.5;">{message}</p>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
