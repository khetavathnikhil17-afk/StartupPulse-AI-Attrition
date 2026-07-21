"""Explainable AI Page – StartupPulse AI Dashboard (Premium)."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from tensorflow import keras

from dashboard.components.reusable_widgets import (
    section_header, section_sub, loading_spinner,
)

_FIGURES = _ROOT / "reports" / "figures"


def _load_html(name: str) -> str:
    path = _FIGURES / name
    if not path.exists():
        return f"<p style='color:#64748B;'>Figure not found: {name}</p>"
    return path.read_text(encoding="utf-8")


def render() -> None:
    """Render the Explainable AI page."""

    section_header("Explainable AI — SHAP Analysis")
    section_sub("Understand why the model makes specific predictions using Shapley values")

    # ── Global Feature Importance ────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(37,99,235,0.12); color:#60A5FA;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Global</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Feature Importance</span>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            Average impact on model output across all predictions.
        </p>
        """,
        unsafe_allow_html=True,
    )
    html = _load_html("shap_global_feature_importance.html")
    st.components.v1.html(html, height=800, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Summary Plot ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(124,58,237,0.12); color:#A78BFA;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Beeswarm</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Summary Plot</span>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            Each dot = one employee. Colour = feature value. Position = SHAP impact.
        </p>
        """,
        unsafe_allow_html=True,
    )
    html = _load_html("shap_summary_plot.html")
    st.components.v1.html(html, height=900, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Local Explanation ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(6,182,212,0.12); color:#22D3EE;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Local</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Prediction Explanation</span>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            How each feature contributed to a <strong style="color:#CBD5E1;">single employee's</strong> prediction.
        </p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">📊 Feature Importance (Local)</div>',
            unsafe_allow_html=True,
        )
        html = _load_html("shap_local_feature_importance_sample_0.html")
        st.components.v1.html(html, height=800, scrolling=True)
    with c2:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">📉 Waterfall Plot</div>',
            unsafe_allow_html=True,
        )
        html = _load_html("shap_waterfall_sample_0.html")
        st.components.v1.html(html, height=800, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Force Plot ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(245,158,11,0.12); color:#FBBF24;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Force</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Force Plot</span>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            Visual breakdown of positive (red) and negative (blue) feature contributions.
        </p>
        """,
        unsafe_allow_html=True,
    )
    html = _load_html("shap_force_plot_sample_0.html")
    st.components.v1.html(html, height=450, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Top Positive / Negative Features ─────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(34,197,94,0.12); color:#4ADE80;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Ranking</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Top Positive vs Negative Features</span>
        </div>
        <p style="color:#64748B; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            Features that push predictions <strong style="color:#F87171;">towards attrition</strong>
            vs <strong style="color:#60A5FA;">away from attrition</strong>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_data
    def _compute_top_features():
        train = pd.read_csv(_ROOT / "data" / "processed" / "train.csv")
        feats = [c for c in train.columns if c != "Attrition"]
        X_bg = train[feats].values[:100].astype(np.float32)
        X_test = train[feats].values[:50].astype(np.float32)
        model = keras.models.load_model(_ROOT / "models" / "startuppulse_v1" / "attrition_model.keras")
        explainer = shap.DeepExplainer(model, X_bg)
        raw = explainer.shap_values(X_test)
        sv = raw[0] if isinstance(raw, list) else raw
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        return feats, sv

    feats, sv = _compute_top_features()
    mean_pos = np.where(sv.mean(axis=0) > 0, sv.mean(axis=0), 0)
    mean_neg = np.where(sv.mean(axis=0) < 0, sv.mean(axis=0), 0)
    order_pos = np.argsort(mean_pos)[::-1][:10]
    order_neg = np.argsort(mean_neg)[:10]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            y=[feats[i] for i in order_pos][::-1],
            x=[mean_pos[i] for i in order_pos][::-1],
            orientation="h",
            marker=dict(color="#EF4444", line=dict(width=0)),
            text=[f"{mean_pos[i]:.4f}" for i in order_pos][::-1],
            textposition="outside",
            textfont=dict(color="#CBD5E1", size=11),
        ))
        fig.update_layout(
            title=dict(text="↑ Increases Attrition Risk", font=dict(color="#F87171", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"), height=420,
            margin=dict(l=180, r=30, t=50, b=30),
            xaxis=dict(gridcolor="rgba(148,163,184,0.06)", zeroline=False),
            yaxis=dict(gridcolor="rgba(148,163,184,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(
            y=[feats[i] for i in order_neg][::-1],
            x=[mean_neg[i] for i in order_neg][::-1],
            orientation="h",
            marker=dict(color="#3B82F6", line=dict(width=0)),
            text=[f"{mean_neg[i]:.4f}" for i in order_neg][::-1],
            textposition="outside",
            textfont=dict(color="#CBD5E1", size=11),
        ))
        fig.update_layout(
            title=dict(text="↓ Decreases Attrition Risk", font=dict(color="#60A5FA", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"), height=420,
            margin=dict(l=180, r=30, t=50, b=30),
            xaxis=dict(gridcolor="rgba(148,163,184,0.06)", zeroline=False),
            yaxis=dict(gridcolor="rgba(148,163,184,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)
