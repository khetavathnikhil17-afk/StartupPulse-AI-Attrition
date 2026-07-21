"""Predict Attrition Page – StartupPulse AI Dashboard (Premium)."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dashboard.components.reusable_widgets import (
    metric_card, risk_badge, risk_meter,
    section_header, section_sub,
)

# ---------------------------------------------------------------------------
# Feature definitions grouped by category
# ---------------------------------------------------------------------------
_PERSONAL = {
    "Age":                      {"type": "number", "min": 18, "max": 65, "default": 35},
    "Gender":                   {"type": "select", "options": ["Female", "Male"], "default": "Male"},
    "MaritalStatus":            {"type": "select", "options": ["Divorced", "Married", "Single"], "default": "Married"},
    "Education":                {"type": "select", "options": [1, 2, 3, 4, 5], "default": 3},
    "EducationField":           {"type": "select", "options": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"], "default": "Life Sciences"},
    "DistanceFromHome":         {"type": "number", "min": 0, "max": 50, "default": 5},
}

_JOB = {
    "Department":               {"type": "select", "options": ["Human Resources", "Research & Development", "Sales"], "default": "Research & Development"},
    "JobRole":                  {"type": "select", "options": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative",
    ], "default": "Research Scientist"},
    "JobLevel":                 {"type": "select", "options": [1, 2, 3, 4, 5], "default": 2},
    "JobInvolvement":           {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "JobSatisfaction":          {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "BusinessTravel":           {"type": "select", "options": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"], "default": "Travel_Rarely"},
}

_COMPENSATION = {
    "MonthlyIncome":            {"type": "number", "min": 1000, "max": 30000, "default": 6000},
    "MonthlyRate":              {"type": "number", "min": 1000, "max": 30000, "default": 15000},
    "DailyRate":                {"type": "number", "min": 100, "max": 1500, "default": 800},
    "HourlyRate":               {"type": "number", "min": 20, "max": 150, "default": 65},
    "PercentSalaryHike":        {"type": "number", "min": 5, "max": 50, "default": 15},
    "StockOptionLevel":         {"type": "select", "options": [0, 1, 2, 3], "default": 1},
}

_ENVIRONMENT = {
    "EnvironmentSatisfaction":  {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "WorkLifeBalance":          {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "RelationshipSatisfaction": {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "PerformanceRating":        {"type": "select", "options": [1, 2, 3, 4], "default": 3},
    "OverTime":                 {"type": "select", "options": ["No", "Yes"], "default": "No"},
    "TrainingTimesLastYear":    {"type": "number", "min": 0, "max": 10, "default": 2},
}

_TENURE = {
    "TotalWorkingYears":        {"type": "number", "min": 0, "max": 40, "default": 8},
    "NumCompaniesWorked":       {"type": "number", "min": 0, "max": 10, "default": 2},
    "YearsAtCompany":           {"type": "number", "min": 0, "max": 40, "default": 5},
    "YearsInCurrentRole":       {"type": "number", "min": 0, "max": 20, "default": 3},
    "YearsSinceLastPromotion":  {"type": "number", "min": 0, "max": 15, "default": 1},
    "YearsWithCurrManager":     {"type": "number", "min": 0, "max": 20, "default": 4},
}


def _render_fields(fields: dict, cols=3) -> dict:
    """Render a group of fields in a grid."""
    values = {}
    items = list(fields.items())
    for row_start in range(0, len(items), cols):
        columns = st.columns(cols)
        for i, c in enumerate(columns):
            idx = row_start + i
            if idx >= len(items):
                break
            fname, spec = items[idx]
            with c:
                if spec["type"] == "number":
                    values[fname] = st.number_input(
                        fname, min_value=spec["min"],
                        max_value=spec["max"], value=spec["default"],
                        label_visibility="visible",
                    )
                else:
                    values[fname] = st.selectbox(
                        fname, options=spec["options"],
                        index=spec["options"].index(spec["default"]),
                        label_visibility="visible",
                    )
    return values


def render() -> None:
    """Render the Predict Attrition page."""

    section_header("Predict Employee Attrition")
    section_sub("Fill in the employee details below to get an instant risk assessment")

    # ── Tabs for organized form ───────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤  Personal",
        "💼  Job",
        "💰  Compensation",
        "🏢  Environment",
        "📅  Tenure",
    ])

    employee_data = {}
    with tab1:
        employee_data.update(_render_fields(_PERSONAL))
    with tab2:
        employee_data.update(_render_fields(_JOB))
    with tab3:
        employee_data.update(_render_fields(_COMPENSATION))
    with tab4:
        employee_data.update(_render_fields(_ENVIRONMENT))
    with tab5:
        employee_data.update(_render_fields(_TENURE))

    st.markdown("")

    # ── Predict Button ────────────────────────────────────────────────
    predict_clicked = st.button(
        "🔍  Analyze Attrition Risk",
        use_container_width=True,
        type="primary",
    )

    if predict_clicked:
        with st.spinner("Analyzing employee profile..."):
            try:
                from src.model.predict import predict_attrition
                result = predict_attrition(employee_data)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.markdown("")

        # ── Result Card ───────────────────────────────────────────────
        is_leave = result.prediction == "Likely to Leave"
        border_color = "rgba(239,68,68,0.3)" if is_leave else "rgba(34,197,94,0.3)"
        glow_color = "rgba(239,68,68,0.08)" if is_leave else "rgba(34,197,94,0.08)"

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {glow_color}, transparent 60%);
                border: 1px solid {border_color};
                border-radius: 20px;
                padding: 2.5rem 2rem;
                text-align: center;
                margin-bottom: 1.5rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 0.6rem;">
                    {'⚠️' if is_leave else '✅'}
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.5px;">
                    {result.prediction}
                </div>
                <div style="margin-top: 0.6rem;">{risk_badge(result.risk_level)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Risk Meter ────────────────────────────────────────────────
        st.markdown(risk_meter(result.probability), unsafe_allow_html=True)
        st.markdown("")

        # ── Metric Cards ──────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        prob_color = "#EF4444" if result.probability >= 0.6 else "#F59E0B" if result.probability >= 0.3 else "#22C55E"
        conf_color = "#22C55E" if result.confidence >= 0.7 else "#F59E0B"
        risk_color = "#EF4444" if "High" in result.risk_level else "#F59E0B" if "Medium" in result.risk_level else "#22C55E"

        with c1:
            metric_card(f"{result.probability:.1%}", "Probability", prob_color)
        with c2:
            metric_card(f"{result.confidence:.1%}", "Confidence", conf_color)
        with c3:
            metric_card(result.risk_level, "Risk Level", risk_color)
        with c4:
            metric_card("DNN v1", "Model", "#60A5FA")

        st.markdown("")

        # ── Recommended Action ─────────────────────────────────────────
        action_bg = "rgba(239,68,68,0.06)" if "High" in result.risk_level else "rgba(245,158,11,0.06)" if "Medium" in result.risk_level else "rgba(34,197,94,0.06)"
        action_border = "rgba(239,68,68,0.2)" if "High" in result.risk_level else "rgba(245,158,11,0.2)" if "Medium" in result.risk_level else "rgba(34,197,94,0.2)"
        action_color = "#F87171" if "High" in result.risk_level else "#FBBF24" if "Medium" in result.risk_level else "#4ADE80"
        action_icon = "🚨" if "High" in result.risk_level else "⚡" if "Medium" in result.risk_level else "✅"

        st.markdown(
            f"""
            <div style="
                background: var(--glass, rgba(15,23,42,0.55));
                border: 1px solid var(--glass-border, rgba(255,255,255,0.06));
                border-radius: 16px;
                padding: 1.5rem;
            ">
                <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.7rem;">
                    <span style="font-size:1.5rem;">{action_icon}</span>
                    <span style="font-size:1rem; font-weight:700; color:#F1F5F9;">Recommended HR Action</span>
                </div>
                <div style="
                    background: {action_bg};
                    border: 1px solid {action_border};
                    border-radius: 12px;
                    padding: 1rem 1.2rem;
                ">
                    <p style="color:{action_color}; font-size:0.9rem; line-height:1.7; margin:0;">
                        {result.recommended_action}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
