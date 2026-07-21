"""Analytics Dashboard Page – StartupPulse AI Dashboard (Premium)."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.reusable_widgets import (
    default_plotly_layout, kpi_card, section_header, section_sub,
)

_FIGURES = _ROOT / "reports" / "figures"
_LAYOUT = default_plotly_layout()


@st.cache_data
def _load_raw() -> pd.DataFrame:
    path = _ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not path.exists():
        st.error(f"Dataset not found: {path.name}")
        st.stop()
    return pd.read_csv(path)


def render() -> None:
    """Render the Analytics Dashboard page."""

    df = _load_raw()
    total = len(df)
    leavers = len(df[df["Attrition"] == "Yes"])

    section_header("Analytics Dashboard")
    section_sub("Interactive workforce insights powered by Plotly")

    # ── KPI Row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("👥", f"{total:,}", "Total Employees", delay=0)
    with c2: kpi_card("🚪", f"{leavers:,}", "Left Company", delay=60)
    with c3: kpi_card("🎯", f"{leavers/total*100:.1f}%" if total > 0 else "0%", "Attrition Rate", delay=120)
    with c4: kpi_card("💰", f"${df['MonthlyIncome'].median():,.0f}", "Median Income", delay=180)
    with c5: kpi_card("📊", f"{df['Age'].mean():.0f}", "Avg Age", delay=240)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Row 1: Attrition + Department + Gender ───────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        attr_counts = df["Attrition"].value_counts()
        fig = go.Figure(go.Pie(
            labels=attr_counts.index, values=attr_counts.values,
            hole=0.6, marker=dict(colors=["#3B82F6", "#EF4444"]),
            textinfo="label+percent", textfont_size=13,
            textfont=dict(color="#F1F5F9"),
        ))
        fig.update_layout(**_LAYOUT, title="Attrition Distribution", height=380,
                          annotations=[dict(text=f"<b>{leavers}</b><br>Left", x=0.5, y=0.5,
                                           font_size=18, font_color="#F1F5F9", showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ct = pd.crosstab(df["Department"], df["Attrition"], normalize="index") * 100
        fig = go.Figure()
        for col, color in zip(["No", "Yes"], ["#3B82F6", "#EF4444"]):
            fig.add_trace(go.Bar(
                name=col, x=ct.index, y=ct[col].round(1),
                marker_color=color, text=ct[col].round(1), textposition="auto",
                textfont=dict(color="#F1F5F9"),
            ))
        fig.update_layout(**_LAYOUT, barmode="group", title="Department × Attrition (%)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        ct = pd.crosstab(df["Gender"], df["Attrition"], normalize="index") * 100
        fig = go.Figure()
        for col, color in zip(["No", "Yes"], ["#3B82F6", "#EF4444"]):
            fig.add_trace(go.Bar(
                name=col, x=ct.index, y=ct[col].round(1),
                marker_color=color, text=ct[col].round(1), textposition="auto",
                textfont=dict(color="#F1F5F9"),
            ))
        fig.update_layout(**_LAYOUT, barmode="group", title="Gender × Attrition (%)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Age + Income + Job Satisfaction ───────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(
            df, x="Age", color="Attrition", nbins=30, barmode="overlay", opacity=0.75,
            color_discrete_map={"No": "#3B82F6", "Yes": "#EF4444"},
        )
        fig.update_layout(**_LAYOUT, title="Age Distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            df, y="MonthlyIncome", x="Attrition", color="Attrition",
            color_discrete_map={"No": "#3B82F6", "Yes": "#EF4444"},
            labels={"MonthlyIncome": "Monthly Income ($)"},
        )
        fig.update_layout(**_LAYOUT, title="Monthly Income vs Attrition", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        sat_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
        tmp = df.copy()
        tmp["JobSat"] = tmp["JobSatisfaction"].map(sat_map)
        ct = pd.crosstab(tmp["JobSat"], tmp["Attrition"], normalize="index").reindex(
            ["Low", "Medium", "High", "Very High"]
        ) * 100
        fig = go.Figure(go.Scatter(
            x=ct.index, y=ct["Yes"].round(1), mode="lines+markers+text",
            line=dict(color="#EF4444", width=3), marker=dict(size=10, color="#EF4444"),
            text=ct["Yes"].round(1).tolist(), textposition="top center",
            textfont=dict(color="#F1F5F9"),
        ))
        fig.update_layout(**_LAYOUT, title="Job Satisfaction → Attrition (%)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Years at Company + Job Role ───────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        fig = px.violin(
            df, y="YearsAtCompany", x="Attrition", color="Attrition", box=True,
            color_discrete_map={"No": "#3B82F6", "Yes": "#EF4444"},
        )
        fig.update_layout(**_LAYOUT, title="Years at Company Distribution", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ct = pd.crosstab(df["JobRole"], df["Attrition"]).sort_values("Yes", ascending=True)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Stayed", y=ct_pct.index, x=ct_pct["No"],
                             orientation="h", marker_color="#3B82F6"))
        fig.add_trace(go.Bar(name="Left", y=ct_pct.index, x=ct_pct["Yes"],
                             orientation="h", marker_color="#EF4444"))
        fig.update_layout(**_LAYOUT, barmode="stack", title="Job Role Attrition (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    section_header("Correlation Heatmap")
    section_sub("Pearson correlations across all numeric features")

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig = go.Figure(go.Heatmap(
        z=corr_masked.values, x=corr_masked.columns, y=corr_masked.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr_masked.values, 2), texttemplate="%{text}",
        textfont={"size": 9, "color": "#CBD5E1"},
    ))
    fig.update_layout(**_LAYOUT, title="Feature Correlation Matrix", height=820, width=920)
    st.plotly_chart(fig, use_container_width=True)
