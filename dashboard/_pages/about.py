"""About Page – StartupPulse AI Dashboard (Premium)."""

import streamlit as st
from dashboard.components.reusable_widgets import section_header, section_sub


def render() -> None:
    """Render the About page."""

    section_header("About StartupPulse AI")
    section_sub("Enterprise HR analytics platform built for scale")

    # ── Project Description ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                <span style="font-size:1.4rem;">🎯</span>
                <h3 style="margin:0; font-size:1.05rem;">Project Description</h3>
            </div>
            <p style="color:#94A3B8; line-height:1.8; font-size:0.9rem; margin:0;">
                StartupPulse AI is a production-ready Employee Attrition Prediction system
                that combines <strong style="color:#F1F5F9;">Deep Learning</strong>,
                <strong style="color:#F1F5F9;">Explainable AI (SHAP)</strong>, and
                <strong style="color:#F1F5F9;">Interactive Analytics</strong> into a
                premium Streamlit dashboard. It enables HR teams to proactively identify
                flight risks, understand the drivers behind attrition, and take
                targeted retention action — all in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Technology Stack ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
                <span style="font-size:1.4rem;">🛠️</span>
                <h3 style="margin:0; font-size:1.05rem;">Technology Stack</h3>
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.8rem;">
        """,
        unsafe_allow_html=True,
    )

    techs = [
        ("🧠", "TensorFlow / Keras", "Deep Learning Framework", "rgba(37,99,235,0.1)", "rgba(37,99,235,0.2)", "#60A5FA"),
        ("🔍", "SHAP 0.49", "Model Explainability", "rgba(124,58,237,0.1)", "rgba(124,58,237,0.2)", "#A78BFA"),
        ("📊", "Plotly 6.9", "Interactive Visualizations", "rgba(6,182,212,0.1)", "rgba(6,182,212,0.2)", "#22D3EE"),
        ("🖥️", "Streamlit 1.59", "Web Application", "rgba(34,197,94,0.1)", "rgba(34,197,94,0.2)", "#4ADE80"),
        ("📦", "scikit-learn 1.7", "Preprocessing & Metrics", "rgba(245,158,11,0.1)", "rgba(245,158,11,0.2)", "#FBBF24"),
        ("🐍", "Python 3.10", "Programming Language", "rgba(239,68,68,0.1)", "rgba(239,68,68,0.2)", "#F87171"),
    ]

    for icon, name, desc, bg, border, color in techs:
        st.markdown(
            f"""
            <div style="
                background:{bg}; border:1px solid {border};
                border-radius:14px; padding:1.1rem 0.8rem;
                text-align:center; transition:all 0.3s ease;
            ">
                <div style="font-size:1.8rem; margin-bottom:0.3rem;">{icon}</div>
                <div style="font-weight:700; color:#F1F5F9; font-size:0.88rem;">{name}</div>
                <div style="font-size:0.72rem; color:#64748B; margin-top:0.15rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("")

    # ── Dataset Information ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
                <span style="font-size:1.4rem;">📊</span>
                <h3 style="margin:0; font-size:1.05rem;">Dataset Information</h3>
            </div>
            <table style="width:100%; border-collapse:separate; border-spacing:0;">
                <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
                    <td style="padding:0.7rem 0; font-weight:600; color:#CBD5E1; width:180px; font-size:0.88rem;">Source</td>
                    <td style="padding:0.7rem 0; color:#94A3B8; font-size:0.88rem;">IBM HR Analytics Employee Attrition & Performance</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
                    <td style="padding:0.7rem 0; font-weight:600; color:#CBD5E1; font-size:0.88rem;">Rows</td>
                    <td style="padding:0.7rem 0; color:#94A3B8; font-size:0.88rem;">1,470 employees</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
                    <td style="padding:0.7rem 0; font-weight:600; color:#CBD5E1; font-size:0.88rem;">Features</td>
                    <td style="padding:0.7rem 0; color:#94A3B8; font-size:0.88rem;">35 columns (30 used after preprocessing)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
                    <td style="padding:0.7rem 0; font-weight:600; color:#CBD5E1; font-size:0.88rem;">Target</td>
                    <td style="padding:0.7rem 0; color:#94A3B8; font-size:0.88rem;">Attrition (Yes / No → 1 / 0)</td>
                </tr>
                <tr>
                    <td style="padding:0.7rem 0; font-weight:600; color:#CBD5E1; font-size:0.88rem;">Class Balance</td>
                    <td style="padding:0.7rem 0; color:#94A3B8; font-size:0.88rem;">83.9% No / 16.1% Yes (imbalanced)</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Developer & Version ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass-card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <span style="font-size:1.4rem;">👨‍💻</span>
                    <h3 style="margin:0; font-size:1.05rem;">Developer</h3>
                </div>
                <div style="color:#94A3B8; font-size:0.88rem; line-height:2.2;">
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Project</span>
                        <span>StartupPulse AI</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Role</span>
                        <span>Senior AI Engineer</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Focus</span>
                        <span>DL, XAI, MLOps</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Status</span>
                        <span style="color:#4ADE80;">● Production v1.0</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                    <span style="font-size:1.4rem;">📦</span>
                    <h3 style="margin:0; font-size:1.05rem;">Version & License</h3>
                </div>
                <div style="color:#94A3B8; font-size:0.88rem; line-height:2.2;">
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Version</span>
                        <span>1.0.0</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Python</span>
                        <span>3.10+</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(148,163,184,0.06); padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">License</span>
                        <span>MIT</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:0.2rem 0;">
                        <span style="color:#CBD5E1; font-weight:600;">Updated</span>
                        <span>July 2026</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
