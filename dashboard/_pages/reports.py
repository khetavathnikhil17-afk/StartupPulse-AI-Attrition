"""Reports Page – StartupPulse AI Dashboard (Premium)."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard.components.reusable_widgets import (
    metric_card, section_header, section_sub, load_metrics,
)

_FIGURES = _ROOT / "reports" / "figures"


def _load_html(name: str) -> str:
    path = _FIGURES / name
    if not path.exists():
        return f"<p style='color:#64748B;'>Figure not found: {name}</p>"
    return path.read_text(encoding="utf-8")


def render() -> None:
    """Render the Reports page."""

    section_header("Model Evaluation Reports")
    section_sub("Comprehensive performance metrics and visual diagnostics")

    metrics = load_metrics()

    # ── Animated Metric Cards ────────────────────────────────────────────
    if metrics:
        st.markdown("")
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            ("🎯", f"{metrics.get('accuracy',0):.1%}", "Accuracy", "#60A5FA"),
            ("🔍", f"{metrics.get('precision',0):.1%}", "Precision", "#22D3EE"),
            ("📡", f"{metrics.get('recall',0):.1%}", "Recall", "#4ADE80"),
            ("⚖️", f"{metrics.get('f1_score',0):.1%}", "F1 Score", "#A78BFA"),
            ("📈", f"{metrics.get('roc_auc',0):.1%}", "ROC AUC", "#FBBF24"),
        ]
        for col, (icon, val, label, color) in zip([c1, c2, c3, c4, c5], cards):
            with col:
                # Colored top accent
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="
                            position:absolute; top:0; left:0; right:0; height:2px;
                            background: {color}; border-radius:16px 16px 0 0;
                        "></div>
                        <div style="font-size:1.4rem; margin-bottom:0.3rem;">{icon}</div>
                        <div class="metric-value" style="color:{color}; font-size:1.6rem;">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Confusion Matrix ─────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(37,99,235,0.12); color:#60A5FA;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Diagnostic</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Confusion Matrix</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    html = _load_html("confusion_matrix.html")
    st.components.v1.html(html, height=500, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Training Curves ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(124,58,237,0.12); color:#A78BFA;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Training</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Training Curves</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">📈 Accuracy Curve</div>',
            unsafe_allow_html=True,
        )
        html = _load_html("training_accuracy_curve.html")
        st.components.v1.html(html, height=500, scrolling=True)
    with c2:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">📉 Loss Curve</div>',
            unsafe_allow_html=True,
        )
        html = _load_html("training_loss_curve.html")
        st.components.v1.html(html, height=500, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Full Evaluation Report ───────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="
                background:rgba(6,182,212,0.12); color:#22D3EE;
                border-radius:8px; padding:0.2rem 0.6rem;
                font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
            ">Report</span>
            <span style="font-size:1.05rem; font-weight:700; color:#F1F5F9;">Full Evaluation Report</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    report_path = _ROOT / "reports" / "results" / "evaluation_report.md"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        st.markdown(
            f"""
            <div class="glass-card" style="max-height:600px; overflow-y:auto; padding:1.5rem;">
                <pre style="
                    color:#94A3B8; white-space:pre-wrap;
                    font-family:'JetBrains Mono','Inter',monospace;
                    font-size:0.82rem; line-height:1.7;
                    margin:0;
                ">{content}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Evaluation report not found.")
