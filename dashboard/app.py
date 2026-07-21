"""
StartupPulse AI – Premium Enterprise Streamlit Dashboard.

Main entry point. Injects the premium dark-theme CSS, renders the
sidebar navigation, and delegates to the active page module.
"""

import importlib
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dashboard.components.reusable_widgets import inject_global_css

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — hide all default Streamlit chrome
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StartupPulse AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject premium CSS
# ---------------------------------------------------------------------------
inject_global_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = {
    "🏠  Home": "home",
    "📈  Predict Attrition": "predict",
    "📊  Analytics Dashboard": "analytics",
    "🔍  Explainable AI": "explainability",
    "📋  Reports": "reports",
    "ℹ️  About": "about",
}

with st.sidebar:
    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brand-icon">🧠</span>
            <span class="sidebar-brand-text">StartupPulse AI</span>
            <div class="sidebar-brand-sub">Enterprise HR Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation
    selected = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    # Version footer
    st.markdown(
        '<div class="sidebar-version">v1.0.0 · Built with TensorFlow + SHAP</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Render active page
# ---------------------------------------------------------------------------
page_key = PAGES[selected]
try:
    page_module = importlib.import_module(f"dashboard._pages.{page_key}")
    page_module.render()
except ImportError as exc:
    logger.error("Failed to import page '%s': %s", page_key, exc)
    st.error(f"Failed to load page: {page_key}")
except Exception as exc:
    logger.error("Error rendering page '%s': %s", page_key, exc)
    st.error(f"Error rendering page: {exc}")
