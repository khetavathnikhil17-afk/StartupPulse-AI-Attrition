"""Home Page – StartupPulse AI Dashboard (Premium Enterprise)."""

import json
import logging
from pathlib import Path

import streamlit as st
from dashboard.components.reusable_widgets import (
    kpi_card, nav_card, section_header, section_sub,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]


def _load_kpis() -> list:
    """Load KPI values from metrics.json and dataset, with fallback defaults."""
    defaults = [
        ("📊", "—", "Total Employees"),
        ("🎯", "—", "Model Accuracy"),
        ("📈", "—", "ROC AUC Score"),
        ("⚡", "—", "Recall Score"),
        ("🏢", "—", "Departments"),
    ]
    try:
        metrics_path = _ROOT / "reports" / "results" / "metrics.json"
        raw_path = _ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

        accuracy = "—"
        roc_auc = "—"
        recall = "—"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            accuracy = f"{m.get('accuracy', 0) * 100:.1f}%"
            roc_auc = f"{m.get('roc_auc', 0) * 100:.1f}%"
            recall = f"{m.get('recall', 0) * 100:.1f}%"

        total = "—"
        departments = "—"
        if raw_path.exists():
            import pandas as pd
            df = pd.read_csv(raw_path)
            total = f"{len(df):,}"
            departments = str(df["Department"].nunique())

        return [
            ("📊", total, "Total Employees"),
            ("🎯", accuracy, "Model Accuracy"),
            ("📈", roc_auc, "ROC AUC Score"),
            ("⚡", recall, "Recall Score"),
            ("🏢", departments, "Departments"),
        ]
    except Exception as exc:
        logger.warning("Failed to load KPIs dynamically: %s", exc)
        return defaults


def render() -> None:
    """Render the Home page."""

    # ══════════════════════════════════════════════════════════════════════
    # HERO SECTION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="'
        'background:linear-gradient(135deg,rgba(15,23,42,0.95) 0%,rgba(30,41,59,0.7) 50%,rgba(15,23,42,0.95) 100%);'
        'border:1px solid rgba(255,255,255,0.06);'
        'border-radius:24px;'
        'padding:4.5rem 3rem 3.5rem;'
        'text-align:center;'
        'margin-bottom:2.5rem;'
        'position:relative;'
        'overflow:hidden;'
        'animation:fadeIn 0.6s ease-out;'
        'box-shadow:0 8px 40px rgba(0,0,0,0.25),inset 0 1px 0 rgba(255,255,255,0.04);'
        '">'
        '<div style="position:absolute;top:-100px;right:-100px;width:350px;height:350px;background:radial-gradient(circle,rgba(37,99,235,0.12),transparent 70%);border-radius:50%;pointer-events:none;"></div>'
        '<div style="position:absolute;bottom:-80px;left:-80px;width:300px;height:300px;background:radial-gradient(circle,rgba(124,58,237,0.10),transparent 70%);border-radius:50%;pointer-events:none;"></div>'
        '<div style="position:absolute;top:30%;left:50%;width:250px;height:250px;background:radial-gradient(circle,rgba(6,182,212,0.06),transparent 70%);border-radius:50%;pointer-events:none;transform:translateX(-50%);"></div>'
        '<div style="position:relative;z-index:1;font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;">'
        '<div style="font-size:3.5rem;margin-bottom:1rem;animation:float 4s ease-in-out infinite;filter:drop-shadow(0 0 24px rgba(37,99,235,0.35));">\U0001f9e0</div>'
        '<h1 style="font-size:3.2rem;font-weight:900;margin:0;line-height:1.05;background:linear-gradient(135deg,#60A5FA 0%,#06B6D4 35%,#A78BFA 65%,#F472B6 100%);background-size:200% 200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:gradientShift 5s ease infinite;letter-spacing:-1.5px;font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;">StartupPulse AI</h1>'
        '<p style="font-size:1.05rem;color:#94A3B8;margin:1rem auto 0;max-width:560px;line-height:1.75;font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;">'
        'Enterprise-grade <strong style="color:#F1F5F9;">Employee Attrition Prediction</strong> powered by Deep Learning, Explainable AI, and Real-Time Analytics.</p>'
        '<div style="display:flex;gap:0.7rem;justify-content:center;margin-top:1.8rem;flex-wrap:wrap;">'
        '<span style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);border-radius:9999px;padding:0.4rem 1.1rem;font-size:0.72rem;color:#60A5FA;font-weight:600;font-family:Inter,sans-serif;letter-spacing:0.3px;">TensorFlow</span>'
        '<span style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);border-radius:9999px;padding:0.4rem 1.1rem;font-size:0.72rem;color:#A78BFA;font-weight:600;font-family:Inter,sans-serif;letter-spacing:0.3px;">SHAP</span>'
        '<span style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:9999px;padding:0.4rem 1.1rem;font-size:0.72rem;color:#FBBF24;font-weight:600;font-family:Inter,sans-serif;letter-spacing:0.3px;">Plotly</span>'
        '<span style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:9999px;padding:0.4rem 1.1rem;font-size:0.72rem;color:#34D399;font-weight:600;font-family:Inter,sans-serif;letter-spacing:0.3px;">Streamlit</span>'
        '</div>'
        '<div style="margin-top:1.5rem;"><span style="font-size:0.65rem;color:#64748B;font-family:JetBrains Mono,monospace;letter-spacing:0.5px;">v1.0.0 | 30 Features | 52K Parameters</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # KPIs
    # ══════════════════════════════════════════════════════════════════════
    section_header("Key Performance Indicators")
    section_sub("Real-time model and dataset metrics")

    cols = st.columns(5)
    kpis = _load_kpis()
    for col, (icon, value, label) in zip(cols, kpis):
        with col:
            kpi_card(icon, value, label)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # PROJECT OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    section_header("Project Overview")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <span style="font-size:1.4rem;">🎯</span>
                    <h3 style="margin:0; font-size:1.05rem;">Mission</h3>
                </div>
                <p style="color:#94A3B8; line-height:1.75; font-size:0.9rem; margin:0;">
                    StartupPulse AI is a production-ready employee attrition prediction
                    system that leverages deep learning and explainable AI to help HR teams
                    proactively identify flight risks and take targeted retention action —
                    before it's too late.
                </p>
                <div style="height:1rem;"></div>
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <span style="font-size:1.4rem;">🚀</span>
                    <h3 style="margin:0; font-size:1.05rem;">Capabilities</h3>
                </div>
                <div style="color:#94A3B8; line-height:2; font-size:0.88rem;">
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="color:#22C55E;">✓</span>
                        <span>Real-time attrition risk scoring with probability estimates</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="color:#22C55E;">✓</span>
                        <span>SHAP-powered explainability for every prediction</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="color:#22C55E;">✓</span>
                        <span>Interactive analytics dashboard with 12+ visualizations</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="color:#22C55E;">✓</span>
                        <span>Automated model evaluation and reporting pipeline</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <span style="font-size:1.4rem;">🏗️</span>
                    <h3 style="margin:0; font-size:1.05rem;">Architecture</h3>
                </div>
                <div style="color:#94A3B8; font-size:0.88rem;">
                    <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(148,163,184,0.06);">
                        <span style="
                            background:rgba(37,99,235,0.15); color:#60A5FA;
                            border-radius:8px; padding:0.2rem 0.5rem;
                            font-size:0.75rem; font-weight:600; min-width:70px; text-align:center;
                        ">Layer 1</span>
                        <span><strong style="color:#F1F5F9;">Data</strong> — Preprocessing & Feature Engineering</span>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(148,163,184,0.06);">
                        <span style="
                            background:rgba(124,58,237,0.15); color:#A78BFA;
                            border-radius:8px; padding:0.2rem 0.5rem;
                            font-size:0.75rem; font-weight:600; min-width:70px; text-align:center;
                        ">Layer 2</span>
                        <span><strong style="color:#F1F5F9;">Model</strong> — Deep Neural Network (52K params)</span>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(148,163,184,0.06);">
                        <span style="
                            background:rgba(6,182,212,0.15); color:#22D3EE;
                            border-radius:8px; padding:0.2rem 0.5rem;
                            font-size:0.75rem; font-weight:600; min-width:70px; text-align:center;
                        ">Layer 3</span>
                        <span><strong style="color:#F1F5F9;">XAI</strong> — SHAP Explainability Engine</span>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(148,163,184,0.06);">
                        <span style="
                            background:rgba(34,197,94,0.15); color:#4ADE80;
                            border-radius:8px; padding:0.2rem 0.5rem;
                            font-size:0.75rem; font-weight:600; min-width:70px; text-align:center;
                        ">Layer 4</span>
                        <span><strong style="color:#F1F5F9;">Analytics</strong> — Plotly Visualizations</span>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.5rem 0;">
                        <span style="
                            background:rgba(245,158,11,0.15); color:#FBBF24;
                            border-radius:8px; padding:0.2rem 0.5rem;
                            font-size:0.75rem; font-weight:600; min-width:70px; text-align:center;
                        ">Layer 5</span>
                        <span><strong style="color:#F1F5F9;">Interface</strong> — Streamlit Dashboard</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW
    # ══════════════════════════════════════════════════════════════════════
    section_header("Workflow")
    section_sub("End-to-end ML pipeline from data to deployment")

    steps = [
        ("📥", "Ingestion", "Load & validate"),
        ("⚙️", "Preprocessing", "Encode & scale"),
        ("🧠", "Training", "DNN + callbacks"),
        ("📊", "Evaluation", "Metrics & reports"),
        ("🔍", "Explainability", "SHAP analysis"),
        ("🚀", "Prediction", "Real-time scoring"),
    ]

    cols = st.columns(6)
    for i, (col, (icon, title, desc)) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding:1.3rem 0.6rem; animation-delay:{i*80}ms">
                    <div style="font-size:1.8rem; margin-bottom:0.4rem;">{icon}</div>
                    <div style="font-weight:700; font-size:0.82rem; color:#F1F5F9; margin-bottom:0.15rem;">{title}</div>
                    <div style="font-size:0.7rem; color:#64748B;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # NAVIGATION CARDS
    # ══════════════════════════════════════════════════════════════════════
    section_header("Quick Navigation")
    section_sub("Jump to any module")

    nav_items = [
        ("📈", "Predict Attrition", "Score employees in real-time"),
        ("📊", "Analytics Dashboard", "Explore workforce insights"),
        ("🔍", "Explainable AI", "Understand model decisions"),
        ("📋", "Reports", "View performance metrics"),
        ("ℹ️", "About", "Project details & tech stack"),
    ]

    cols = st.columns(5)
    for i, (col, (icon, title, desc)) in enumerate(zip(cols, nav_items)):
        with col:
            nav_card(icon, title, desc, delay=i * 60)
