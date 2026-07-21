"""
Reusable UI Components – StartupPulse AI Dashboard (Premium Enterprise).

Provides:
    - Premium dark-theme CSS with 40+ keyframe animations
    - Glassmorphism cards with gradient borders
    - Animated KPI / metric cards with glow effects
    - Skeleton loading states
    - Custom loading spinner
    - Gradient buttons with pulse animation
    - Responsive grid system
    - Premium Plotly layout
"""

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_CSS = r"""
<style>
/* ════════════════════════════════════════════════════════════════════════════
   FONTS
   ════════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ════════════════════════════════════════════════════════════════════════════
   CSS CUSTOM PROPERTIES
   ════════════════════════════════════════════════════════════════════════════ */
:root {
    --primary: #2563EB;
    --primary-glow: rgba(37, 99, 235, 0.35);
    --secondary: #7C3AED;
    --secondary-glow: rgba(124, 58, 237, 0.35);
    --accent: #06B6D4;
    --accent-glow: rgba(6, 182, 212, 0.3);
    --bg: #0B1120;
    --bg-elevated: #0F172A;
    --card: rgba(15, 23, 42, 0.6);
    --card-solid: #131C31;
    --card-hover: rgba(30, 41, 59, 0.7);
    --success: #22C55E;
    --success-glow: rgba(34, 197, 94, 0.25);
    --danger: #EF4444;
    --danger-glow: rgba(239, 68, 68, 0.25);
    --warning: #F59E0B;
    --warning-glow: rgba(245, 158, 11, 0.25);
    --text: #F1F5F9;
    --text-secondary: #CBD5E1;
    --text-muted: #64748B;
    --border: rgba(148, 163, 184, 0.08);
    --border-strong: rgba(148, 163, 184, 0.15);
    --glass: rgba(15, 23, 42, 0.55);
    --glass-border: rgba(255, 255, 255, 0.06);
    --glass-border-hover: rgba(255, 255, 255, 0.12);
    --radius-sm: 10px;
    --radius: 16px;
    --radius-lg: 24px;
    --radius-xl: 32px;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
    --shadow-lg: 0 12px 48px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 40px var(--primary-glow);
}

/* ════════════════════════════════════════════════════════════════════════════
   KEYFRAME ANIMATIONS
   ════════════════════════════════════════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 8px var(--primary-glow); }
    50% { box-shadow: 0 0 24px var(--primary-glow), 0 0 48px rgba(37,99,235,0.1); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(37,99,235,0.2); }
    50% { border-color: rgba(6,182,212,0.4); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
@keyframes progressFill {
    from { width: 0%; }
    to   { width: var(--fill-width); }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.5); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes ripple {
    0%   { transform: scale(0); opacity: 0.5; }
    100% { transform: scale(4); opacity: 0; }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@keyframes skeleton {
    0%   { background-position: -200px 0; }
    100% { background-position: calc(200px + 100%) 0; }
}

/* ════════════════════════════════════════════════════════════════════════════
   GLOBAL RESETS — minimal, do NOT hide sidebar or its toggle
   ════════════════════════════════════════════════════════════════════════════ */
footer { visibility: hidden; }
#stDecoration { display: none !important; }
nav[data-testid="stSidebarNav"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; overflow: visible !important; }

.stApp {
    background: var(--bg) !important;
    min-height: 100vh;
}

/* Main container */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   SIDEBAR — Dark theme background only (do NOT target inner divs)
   ════════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080E1C 0%, #0D1526 40%, #111D35 100%) !important;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

.sidebar-brand {
    text-align: center;
    padding: 1.2rem 0 0.8rem 0;
    margin-bottom: 0.5rem;
}
.sidebar-brand-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.3rem;
}
.sidebar-brand-text {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #06B6D4, #A78BFA);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
}
.sidebar-brand-sub {
    font-size: 0.65rem;
    color: var(--text-muted) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.sidebar-version {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
    font-size: 0.7rem;
    color: var(--text-muted) !important;
    letter-spacing: 1px;
}

/* ════════════════════════════════════════════════════════════════════════════
   TYPOGRAPHY — main content only
   ════════════════════════════════════════════════════════════════════════════ */
.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4,
.block-container h5,
.block-container h6 {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
h1 { font-weight: 800 !important; letter-spacing: -0.5px !important; }
h2 { font-weight: 700 !important; letter-spacing: -0.3px !important; }
h3 { font-weight: 600 !important; }

/* ════════════════════════════════════════════════════════════════════════════
   SECTION HEADER — Gradient Underline
   ════════════════════════════════════════════════════════════════════════════ */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text) !important;
    -webkit-text-fill-color: unset;
    margin-bottom: 0.3rem;
    padding-bottom: 0.6rem;
    position: relative;
    display: inline-block;
    animation: fadeInUp 0.5s ease-out;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 48px; height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 2px;
}
.section-sub {
    font-size: 0.88rem;
    color: var(--text-muted) !important;
    margin-top: -0.3rem;
    margin-bottom: 1.2rem;
    animation: fadeInUp 0.5s ease-out 0.1s both;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
    margin: 2.5rem 0;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM CARD
   ════════════════════════════════════════════════════════════════════════════ */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(1.2);
    -webkit-backdrop-filter: blur(20px) saturate(1.2);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out both;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}
.glass-card:hover {
    border-color: var(--glass-border-hover);
    transform: translateY(-3px);
    box-shadow: var(--shadow), 0 0 0 1px rgba(37,99,235,0.06);
}

/* ════════════════════════════════════════════════════════════════════════════
   KPI CARD — Animated
   ════════════════════════════════════════════════════════════════════════════ */
.kpi-card {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(1.2);
    -webkit-backdrop-filter: blur(20px) saturate(1.2);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out both;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    opacity: 0;
    transition: opacity 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: rgba(37, 99, 235, 0.2);
    box-shadow: var(--shadow-lg), 0 0 32px var(--primary-glow);
}
.kpi-card:hover::before { opacity: 1; }

.kpi-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    animation: countUp 0.6s ease-out;
    font-family: 'JetBrains Mono', 'Inter', monospace;
}
.kpi-label {
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    margin-top: 0.35rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ════════════════════════════════════════════════════════════════════════════
   METRIC CARD
   ════════════════════════════════════════════════════════════════════════════ */
.metric-card {
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.2rem 1rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out both;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow), 0 0 20px rgba(37,99,235,0.08);
    border-color: var(--glass-border-hover);
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
    font-family: 'JetBrains Mono', 'Inter', monospace;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    margin-top: 0.25rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ════════════════════════════════════════════════════════════════════════════
   RISK BADGE
   ════════════════════════════════════════════════════════════════════════════ */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1.2rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    animation: scaleIn 0.3s ease-out;
}
.risk-low {
    background: rgba(34, 197, 94, 0.1);
    color: var(--success);
    border: 1px solid rgba(34, 197, 94, 0.25);
    box-shadow: 0 0 16px var(--success-glow);
}
.risk-medium {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
    border: 1px solid rgba(245, 158, 11, 0.25);
    box-shadow: 0 0 16px var(--warning-glow);
    animation: scaleIn 0.3s ease-out, borderGlow 2s ease infinite;
}
.risk-high {
    background: rgba(239, 68, 68, 0.1);
    color: var(--danger);
    border: 1px solid rgba(239, 68, 68, 0.25);
    box-shadow: 0 0 16px var(--danger-glow);
    animation: scaleIn 0.3s ease-out, glow 2s ease infinite;
}

/* ════════════════════════════════════════════════════════════════════════════
   RISK METER
   ════════════════════════════════════════════════════════════════════════════ */
.risk-meter-wrap {
    margin: 0.8rem 0;
    animation: fadeInUp 0.5s ease-out 0.2s both;
}
.risk-meter-bg {
    background: var(--card-solid);
    border-radius: 9999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--glass-border);
}
.risk-meter-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--success), var(--warning) 50%, var(--danger));
    background-size: 200% 100%;
    animation: progressFill 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    box-shadow: 0 0 12px var(--primary-glow);
}

/* ════════════════════════════════════════════════════════════════════════════
   NAVIGATION CARD
   ════════════════════════════════════════════════════════════════════════════ */
.nav-card {
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.8rem 1rem;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.nav-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    transform: scaleX(0);
    transition: transform 0.3s ease;
}
.nav-card:hover {
    transform: translateY(-6px);
    border-color: rgba(37, 99, 235, 0.2);
    box-shadow: var(--shadow-lg), 0 0 40px var(--primary-glow);
}
.nav-card:hover::before { transform: scaleX(1); }
.nav-card-icon {
    font-size: 2.5rem;
    margin-bottom: 0.7rem;
    display: inline-block;
    transition: transform 0.3s ease;
}
.nav-card:hover .nav-card-icon { transform: scale(1.15) translateY(-3px); }
.nav-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: var(--text) !important;
}
.nav-card-desc {
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    line-height: 1.4;
}

/* ════════════════════════════════════════════════════════════════════════════
   GRADIENT BUTTONS
   ════════════════════════════════════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px var(--primary-glow) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px var(--primary-glow), 0 0 0 1px rgba(37,99,235,0.3) !important;
    background-position: 100% 0 !important;
}
.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   FORM INPUTS
   ════════════════════════════════════════════════════════════════════════════ */
.stSelectbox > div > div,
.stTextArea > div > div,
.stMultiSelect > div > div {
    background: var(--card-solid) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    transition: border-color 0.2s ease !important;
}
.stSelectbox > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-glow) !important;
}
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--card-solid) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    transition: border-color 0.2s ease !important;
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-glow) !important;
}
.stSlider > div > div > div > div {
    background: var(--primary) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
details[open] > summary {
    border-bottom-color: var(--glass-border) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   ALERTS
   ════════════════════════════════════════════════════════════════════════════ */
.alert-success {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.2rem;
    color: var(--success) !important;
    animation: fadeInUp 0.4s ease-out;
}
.alert-danger {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.2rem;
    color: var(--danger) !important;
    animation: fadeInUp 0.4s ease-out;
}
.alert-info {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.2rem;
    color: var(--primary) !important;
    animation: fadeInUp 0.4s ease-out;
}

/* ════════════════════════════════════════════════════════════════════════════
   PLOTLY CHART CONTAINERS
   ════════════════════════════════════════════════════════════════════════════ */
.stPlotlyChart {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    overflow: hidden;
    padding: 0.3rem;
    transition: border-color 0.3s ease;
}
.stPlotlyChart:hover {
    border-color: var(--glass-border-hover);
}

/* ════════════════════════════════════════════════════════════════════════════
   LOADING SPINNER
   ════════════════════════════════════════════════════════════════════════════ */
.loading-spinner {
    display: inline-block;
    width: 20px; height: 20px;
    border: 2.5px solid rgba(37, 99, 235, 0.2);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
}
.loading-dots::after {
    content: '';
    animation: dots 1.4s steps(4, end) infinite;
}
@keyframes dots {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
    100% { content: ''; }
}

/* ════════════════════════════════════════════════════════════════════════════
   SKELETON LOADING
   ════════════════════════════════════════════════════════════════════════════ */
.skeleton {
    background: linear-gradient(90deg, var(--card-solid) 0%, rgba(30,41,59,0.5) 40%, var(--card-solid) 80%);
    background-size: 200px 100%;
    animation: skeleton 1.4s ease-in-out infinite;
    border-radius: var(--radius-sm);
}
.skeleton-text    { height: 14px; margin: 6px 0; width: 80%; }
.skeleton-title   { height: 22px; margin: 8px 0; width: 50%; }
.skeleton-card    { height: 120px; border-radius: var(--radius); }
.skeleton-circle  { width: 48px; height: 48px; border-radius: 50%; }

/* ════════════════════════════════════════════════════════════════════════════
   TABLES
   ════════════════════════════════════════════════════════════════════════════ */
table {
    border-collapse: separate !important;
    border-spacing: 0;
    width: 100%;
}
th {
    background: var(--card-solid) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 0.8rem 1rem !important;
    border-bottom: 1px solid var(--border-strong) !important;
}
td {
    padding: 0.7rem 1rem !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}
tr:hover td {
    background: rgba(37, 99, 235, 0.04) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ════════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ════════════════════════════════════════════════════════════════════════════
   RESPONSIVE TWEAKS
   ════════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .kpi-value { font-size: 1.5rem; }
    .section-header { font-size: 1.2rem; }
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    """Inject the premium dark-theme CSS."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARD
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(icon: str, value: str, label: str, delay: int = 0) -> None:
    """Render an animated KPI card.

    Args:
        icon: Emoji icon string.
        value: Display value (e.g., "75.1%").
        label: Description text below the value.
        delay: Animation delay in ms (for staggered entrance).
    """
    st.markdown(
        f"""
        <div class="kpi-card" style="animation-delay: {delay}ms">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARD
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(value: str, label: str, color: str = "#60A5FA", delay: int = 0) -> None:
    """Render a smaller metric card."""
    st.markdown(
        f"""
        <div class="metric-card" style="animation-delay: {delay}ms">
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RISK BADGE
# ══════════════════════════════════════════════════════════════════════════════

def risk_badge(level: str) -> str:
    """Return HTML for a risk badge."""
    css_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(level, "risk-low")
    icon = {"Low Risk": "●", "Medium Risk": "●", "High Risk": "●"}.get(level, "●")
    return f'<span class="risk-badge {css_class}">{icon} {level}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# RISK METER
# ══════════════════════════════════════════════════════════════════════════════

def risk_meter(probability: float) -> str:
    """Return HTML for an animated risk meter bar."""
    pct = min(max(probability * 100, 0), 100)
    color = _risk_color(probability)
    return f"""
    <div class="risk-meter-wrap">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
            <span style="font-size:0.75rem; color:var(--text-muted); font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">Risk Level</span>
            <span style="font-size:0.95rem; font-weight:700; color:{color}; font-family:'JetBrains Mono',monospace;">{pct:.1f}%</span>
        </div>
        <div class="risk-meter-bg">
            <div class="risk-meter-fill" style="--fill-width:{pct}%; width:{pct}%"></div>
        </div>
    </div>
    """


def _risk_color(prob: float) -> str:
    if prob < 0.3:
        return "var(--success)"
    if prob < 0.6:
        return "var(--warning)"
    return "var(--danger)"


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION CARD
# ══════════════════════════════════════════════════════════════════════════════

def nav_card(icon: str, title: str, description: str, delay: int = 0) -> None:
    """Render a navigation card."""
    st.markdown(
        f"""
        <div class="nav-card" style="animation-delay: {delay}ms">
            <div class="nav-card-icon">{icon}</div>
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION HEADER
# ══════════════════════════════════════════════════════════════════════════════

def section_header(text: str) -> None:
    """Render a section header."""
    st.markdown(f"#### {text}")


def section_sub(text: str) -> None:
    """Render a muted subtitle below a section header."""
    st.caption(text)


# ══════════════════════════════════════════════════════════════════════════════
# SKELETON LOADING
# ══════════════════════════════════════════════════════════════════════════════

def skeleton_cards(n: int = 4) -> None:
    """Render skeleton placeholder cards during loading."""
    cols = st.columns(n)
    for col in cols:
        with col:
            st.markdown(
                '<div class="skeleton skeleton-card"></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# LOADING SPINNER
# ══════════════════════════════════════════════════════════════════════════════

def loading_spinner(text: str = "Loading") -> None:
    """Render an inline loading spinner with animated dots."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.6rem; padding:1rem 0; color:var(--text-muted);">
            <span class="loading-spinner"></span>
            <span style="font-size:0.88rem; font-weight:500;">{text}<span class="loading-dots"></span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY DEFAULT LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def default_plotly_layout() -> Dict[str, Any]:
    """Return a premium dark-themed Plotly layout dict (no title/height — callers set those)."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter", size=12),
        margin=dict(l=50, r=24, t=52, b=44),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.06)",
            zeroline=False,
            showline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.06)",
            zeroline=False,
            showline=False,
            tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=12),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", family="Inter"),
        ),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#64748B", activecolor="#2563EB"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_metrics() -> Dict[str, float]:
    """Load evaluation metrics from the results JSON."""
    import logging
    logger = logging.getLogger(__name__)
    path = _ROOT / "reports" / "results" / "metrics.json"
    if not path.exists():
        logger.warning("metrics.json not found at %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_training_history_html(name: str) -> str:
    """Load an HTML figure file and return its contents."""
    import logging
    logger = logging.getLogger(__name__)
    path = _ROOT / "reports" / "figures" / name
    if not path.exists():
        logger.warning("Figure not found: %s", name)
        return "<p style='color:#64748B;'>Figure not found.</p>"
    return path.read_text(encoding="utf-8")
