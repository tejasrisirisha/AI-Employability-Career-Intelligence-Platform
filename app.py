"""
AI Employability & Career Intelligence Platform  (Upgraded v2)
=================================================================
Streamlit application.

STRUCTURE:

1. "Student Profile" is a full-width MAIN-SCREEN page (sidebar just has a nav
   link to it). Fill it in, click "Run Employability Assessment", and you're
   taken straight to the full report.

2. "Home / Full Report" renders the ENTIRE original pipeline automatically,
   one section after another, on the main screen (Overview, Explainable AI,
   Skill Gap, Job Role Matching, Resume Analysis, Improvement Plan, What-If
   Simulator, Progress Tracking) — no extra clicks needed.

3. The sidebar's "Explore More" menu contains the AI-powered add-ons:
        - 🤖 Ask Your Mentor        (AI career chatbot — now intent-based,
                                      answers a MUCH wider range of questions)
        - 🎤 Interview Lab          (mock interview practice + instant
                                      feedback + a score trend, replaces the
                                      old Goal Optimizer page)
        - 🏢 Company Readiness      (named-company hiring-bar readiness check)

Run with:  streamlit run app.py
"""

import os
import re
import json
import random
import datetime as dt

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from job_roles import JOB_ROLES, RESUME_SKILL_KEYWORDS

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Employability & Career Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "student_employability_dataset.csv")
LOG_PATH = os.path.join(BASE, "data", "progress_log.csv")

# --------------------------------------------------------------------------------------
# GLOBAL STYLE  —  "Aurora Glass" theme: dark aurora backdrop, glassmorphic
# cards, gradient neon accents, glowing hover states. Distinct from the
# generic light Streamlit look.
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --violet: #8b5cf6;
        --aqua: #22d3ee;
        --amber: #fbbf24;
        --rose: #fb7185;
        --emerald: #34d399;
        --glass: rgba(255,255,255,0.055);
        --glass-border: rgba(255,255,255,0.14);
        --ink: #eef1fb;
        --ink-dim: #a6adc9;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
    p, span, label, div, li { color: var(--ink); }
    .stCaption, [data-testid="stCaptionContainer"] p { color: var(--ink-dim) !important; }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(139,92,246,0.20) 0%, transparent 38%),
            radial-gradient(circle at 92% 18%, rgba(34,211,238,0.16) 0%, transparent 42%),
            radial-gradient(circle at 30% 90%, rgba(251,113,133,0.12) 0%, transparent 45%),
            linear-gradient(180deg, #0b0f1e 0%, #0e1226 45%, #0b0f1e 100%);
        background-attachment: fixed;
    }
    .main .block-container { padding-top: 1.6rem; }

    /* ---------- Glass cards ---------- */
    .metric-card, .company-card {
        background: var(--glass);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 20px; padding: 20px 24px; border: 1px solid var(--glass-border);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .metric-card:hover, .company-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(139,92,246,0.22);
        border-color: rgba(139,92,246,0.45);
    }

    /* ---------- Badges ---------- */
    .badge {
        display:inline-block; padding: 5px 15px; border-radius: 999px;
        font-weight: 700; font-size: 0.8rem; margin-right: 6px; letter-spacing: 0.2px;
    }
    .badge-green { background:rgba(52,211,153,0.15); color:#6ee7b7; border:1px solid rgba(52,211,153,0.45); box-shadow:0 0 14px rgba(52,211,153,0.18);}
    .badge-yellow{ background:rgba(251,191,36,0.15); color:#fcd34d; border:1px solid rgba(251,191,36,0.45); box-shadow:0 0 14px rgba(251,191,36,0.18);}
    .badge-red   { background:rgba(251,113,133,0.15); color:#fda4af; border:1px solid rgba(251,113,133,0.45); box-shadow:0 0 14px rgba(251,113,133,0.18);}
    .badge-purple{ background:rgba(139,92,246,0.18); color:#c4b5fd; border:1px solid rgba(139,92,246,0.5); box-shadow:0 0 14px rgba(139,92,246,0.2);}

    .section-title {
        font-size: 1.55rem; font-weight: 800; margin-top: 0.6rem; margin-bottom: 0.6rem;
        background: linear-gradient(90deg, var(--violet), var(--aqua));
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        border-left: 5px solid var(--violet); padding-left: 14px;
    }
    .pill {
        display:inline-block; background: rgba(139,92,246,0.10); border:1px solid rgba(139,92,246,0.35);
        border-radius: 999px; padding: 6px 13px; margin: 3px; font-size:0.83rem; color:#dcd3ff;
    }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; color: var(--ink) !important; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
    div[data-testid="stMetricLabel"] { color: var(--ink-dim) !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1024 0%, #10142a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---------- Buttons ---------- */
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #8b5cf6 0%, #22d3ee 100%);
        color: #0b0f1e !important;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 16px rgba(139,92,246,0.35);
    }
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 28px rgba(34,211,238,0.4);
    }
    div[data-testid="stButton"] button:active,
    div[data-testid="stFormSubmitButton"] button:active { transform: translateY(0px); }

    /* ---------- Inputs ---------- */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.04) !important; color: var(--ink) !important;
        border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.14) !important;
    }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] { font-weight: 600; border-radius: 8px 8px 0 0; color: var(--ink-dim) !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--aqua) !important; }

    /* ---------- Hero banner (animated aurora gradient) ---------- */
    @keyframes auroraShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-banner {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #4c1d95 0%, #6d28d9 30%, #0ea5b7 65%, #4c1d95 100%);
        background-size: 260% 260%;
        animation: auroraShift 14s ease infinite;
        border-radius: 26px; padding: 38px 40px; margin-bottom: 10px;
        box-shadow: 0 20px 46px rgba(76,29,149,0.35);
        color: #ffffff; border: 1px solid rgba(255,255,255,0.14);
    }
    .hero-banner h1 { color: #ffffff !important; font-size: 2.2rem; margin-bottom: 6px; text-shadow: 0 2px 18px rgba(0,0,0,0.25); }
    .hero-banner p { color: #ece9ff !important; font-size: 1.03rem; margin-bottom: 0; }
    .hero-icons { font-size: 1.4rem; margin-top: 12px; letter-spacing: 8px; opacity: 0.92; }

    /* ---------- How It Works ---------- */
    .hiw-wrap {
        margin: 18px 0 30px 0; padding: 24px 20px 20px 20px;
        background: var(--glass); backdrop-filter: blur(14px);
        border: 1px solid var(--glass-border); border-radius: 22px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.3);
    }
    .hiw-title {
        text-align:center; font-size: 2rem; font-weight: 800; margin-bottom:4px;
        background: linear-gradient(90deg, var(--violet), var(--aqua));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hiw-sub { text-align:center; color: var(--ink-dim); font-size:0.98rem; margin-bottom: 24px; font-weight: 500; }
    .hiw-row { display:flex; align-items:stretch; gap: 8px; margin-bottom: 4px; flex-wrap: nowrap; }
    .hiw-card {
        flex: 1; min-width: 0; background: rgba(255,255,255,0.035); border:1px solid var(--glass-border);
        border-radius: 16px; padding: 16px 13px; position: relative;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .hiw-card:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(139,92,246,0.2); border-color: rgba(139,92,246,0.4); }
    .hiw-num {
        display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px;
        border-radius:50%; background: linear-gradient(135deg, var(--violet), var(--aqua)); color:#0b0f1e;
        font-weight:800; font-size:0.85rem; margin-bottom: 8px;
    }
    .hiw-card-title { font-weight:700; font-size:0.92rem; color: var(--ink); margin-bottom:4px; }
    .hiw-card-desc { font-size:0.78rem; color: var(--ink-dim); line-height:1.3; }
    .hiw-arrow { display:flex; align-items:center; justify-content:center; color: var(--violet); font-size:1.1rem; flex: 0 0 18px; }
    @media (max-width: 900px) { .hiw-row { flex-wrap: wrap; } .hiw-arrow { display:none; } }

    /* ---------- Sidebar navigation ---------- */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 9px 12px; margin-bottom: 4px; width: 100%; transition: all 0.15s ease; cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(139,92,246,0.14); border-color: rgba(139,92,246,0.4);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #22d3ee 100%); border-color: transparent;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #0b0f1e !important; font-weight: 800;
    }

    /* ---------- Report section anchors ---------- */
    .report-divider { margin: 34px 0 18px 0; border: none; border-top: 1px dashed rgba(255,255,255,0.16); }
    .toc-pill {
        display:inline-block; background: rgba(139,92,246,0.12); border:1px solid rgba(139,92,246,0.4); color:#c4b5fd;
        border-radius: 999px; padding: 7px 15px; margin: 3px; font-size:0.82rem; font-weight:700; text-decoration: none;
    }
    .toc-pill:hover { background: rgba(139,92,246,0.25); }

    /* ---------- Mentor chat ---------- */
    .mentor-bubble {
        background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(34,211,238,0.08) 100%);
        border: 1px solid rgba(139,92,246,0.3); border-radius: 14px; padding: 12px 16px;
        margin-bottom: 10px; font-size: 0.95rem; line-height: 1.5;
    }

    /* ---------- Interview Lab ---------- */
    .interview-card {
        background: var(--glass); backdrop-filter: blur(14px); border: 1px solid var(--glass-border);
        border-radius: 20px; padding: 22px 26px; box-shadow: 0 10px 26px rgba(0,0,0,0.3); margin-bottom: 16px;
    }
    .interview-question {
        font-size: 1.15rem; font-weight: 700; color: var(--ink); font-family:'Space Grotesk', sans-serif;
        border-left: 4px solid var(--aqua); padding-left: 14px; margin-bottom: 6px;
    }
    .score-ring {
        display:inline-flex; align-items:center; justify-content:center; width:64px; height:64px;
        border-radius:50%; font-weight:800; font-size:1.1rem; font-family:'Space Grotesk', sans-serif;
        background: conic-gradient(var(--emerald) 0deg, rgba(255,255,255,0.06) 0deg); color:#fff;
    }

    /* ---------- Roadmap-style step rows (reused for company gap lists etc.) ---------- */
    .roadmap-step {
        display:flex; align-items:flex-start; gap: 12px; background: rgba(255,255,255,0.035);
        border:1px solid var(--glass-border); border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;
    }
    .roadmap-num {
        flex: 0 0 30px; height:30px; width:30px; border-radius:50%;
        background: linear-gradient(135deg, var(--violet), var(--aqua)); color:#0b0f1e;
        display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem;
    }
    .goal-hit  { background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.4); border-radius:14px; padding:16px 20px; }
    .goal-miss { background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.4); border-radius:14px; padding:16px 20px; }

    /* ===================================================================
       Hard overrides for native Streamlit components. These exist so the
       app looks identical whether the visitor's OS/browser is set to
       light or dark — Streamlit's own auto-theme would otherwise render
       some native widgets in light colors that clash with our dark
       backdrop and become unreadable. Pair with .streamlit/config.toml
       (base="dark") which stops Streamlit from auto-switching at all.
       =================================================================== */

    /* Alerts: info / success / warning / error boxes */
    div[data-testid="stAlert"] {
        background: rgba(255,255,255,0.055) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span, div[data-testid="stAlert"] div { color: var(--ink) !important; }

    /* Expanders (used for the Student Profile sections) */
    div[data-testid="stExpander"] {
        background: var(--glass); border: 1px solid var(--glass-border); border-radius: 18px;
        backdrop-filter: blur(14px); overflow: hidden; margin-bottom: 16px;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--ink) !important;
        padding: 4px 6px;
    }
    div[data-testid="stExpander"] summary:hover { background: rgba(139,92,246,0.10); }
    div[data-testid="stExpander"] svg { color: var(--aqua) !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.045) !important; border: 1px dashed rgba(255,255,255,0.28) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }

    /* Chat bubbles (Ask Your Mentor) */
    [data-testid="stChatMessage"] {
        background: var(--glass) !important; border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important; backdrop-filter: blur(10px);
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--ink) !important; }
    [data-testid="stChatInput"] textarea { background: rgba(255,255,255,0.05) !important; color: var(--ink) !important; }

    /* Progress bars */
    div[data-testid="stProgress"] { background: rgba(255,255,255,0.10) !important; border-radius: 999px; }
    div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--violet), var(--aqua)) !important; }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] { background-color: var(--aqua) !important; box-shadow: 0 0 10px rgba(34,211,238,0.55); }
    div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.14) !important; }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--glass-border); }

    /* Misc text elements that default to system-theme colors */
    a { color: var(--aqua) !important; }
    code { color: #fda4af !important; background: rgba(255,255,255,0.08) !important; border-radius: 6px; }
    hr { border-color: rgba(255,255,255,0.14) !important; }
    [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .stSpinner > div { color: var(--ink) !important; }
    [data-testid="stForm"] { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 22px; padding: 6px 4px; }
    [data-testid="stHeader"] { background: rgba(11,15,30,0.0) !important; }
    [data-testid="stToolbar"] { background: transparent !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_FONT_COLOR = "#eef1fb"
CHART_PAPER_BG = "rgba(0,0,0,0)"
CHART_PLOT_BG = "rgba(0,0,0,0)"

# --------------------------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(MODEL_DIR, "placement_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "package_regressor.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return clf, reg, scaler, encoders, feature_columns, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


clf, reg, scaler, encoders, FEATURE_COLUMNS, METRICS = load_artifacts()
df_reference = load_dataset()

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in ["Gender", "Branch", "Placement_Training"]]
CATEGORICAL_FEATURES = ["Gender", "Branch", "Placement_Training"]

# Try to load SHAP for true explainability; fall back to model feature_importances_
try:
    import shap
    SHAP_AVAILABLE = True
    explainer = shap.TreeExplainer(clf)
except Exception:
    SHAP_AVAILABLE = False
    explainer = None

# --------------------------------------------------------------------------------------
# CORE HELPER FUNCTIONS (prediction / explainability / roles / resume / plan / logging)
# --------------------------------------------------------------------------------------
def build_feature_row(profile: dict) -> pd.DataFrame:
    """Turn a student profile dict into a properly-ordered, encoded feature row."""
    row = {}
    for col in NUMERIC_FEATURES:
        row[col] = profile[col]
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        val = profile[col]
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])[0]
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def compute_employability_score(profile: dict) -> float:
    """0-100 transparent employability score (separate from the ML placement probability)."""
    score = (
        profile["CGPA"] / 10 * 20 + min(profile["Internships"], 3) / 3 * 12 +
        min(profile["Projects"], 5) / 5 * 10 + min(profile["Certifications"], 4) / 4 * 6 +
        profile["Aptitude_Score"] / 100 * 14 + profile["Technical_Skill"] / 10 * 12 +
        profile["Coding_Skill"] / 10 * 8 + profile["Communication_Skill"] / 10 * 8 +
        profile["Soft_Skill"] / 10 * 4 + (3 if profile["Placement_Training"] == "Yes" else 0) +
        profile["Extracurricular_Score"] / 10 * 2 + profile["Leadership_Score"] / 10 * 1 -
        profile["Backlogs"] * 4.2
    )
    return float(np.clip(score, 0, 100))


def predict_profile(profile: dict):
    """Returns (placed_label, placement_probability, predicted_package)"""
    X = build_feature_row(profile)
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0, 1]
    label = int(proba >= 0.5)
    package = float(reg.predict(X_scaled)[0]) if proba >= 0.35 else 0.0
    return label, float(proba), max(package, 0.0)


def get_explanation(profile: dict, top_n=8):
    """Return a DataFrame of top contributing features for this specific student."""
    X = build_feature_row(profile)
    X_scaled = scaler.transform(X)

    if SHAP_AVAILABLE:
        try:
            shap_values = explainer.shap_values(X_scaled)
            vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            contrib = dict(zip(FEATURE_COLUMNS, vals))
            expl_df = pd.DataFrame({
                "Feature": list(contrib.keys()),
                "Impact": list(contrib.values()),
            })
            expl_df["AbsImpact"] = expl_df["Impact"].abs()
            expl_df = expl_df.sort_values("AbsImpact", ascending=False).head(top_n)
            expl_df["Direction"] = np.where(expl_df["Impact"] > 0, "Increases", "Decreases")
            return expl_df[["Feature", "Impact", "Direction"]], "SHAP (exact model attribution)"
        except Exception:
            pass

    # Fallback: global feature importance x local deviation from dataset mean (signed heuristic)
    importances = METRICS["feature_importances"]
    means = df_reference[NUMERIC_FEATURES].mean().to_dict()
    stds = df_reference[NUMERIC_FEATURES].std().to_dict()
    rows = []
    for feat in FEATURE_COLUMNS:
        imp = importances.get(feat, 0.01)
        if feat in NUMERIC_FEATURES:
            z = (profile[feat] - means[feat]) / (stds[feat] + 1e-6)
            if feat == "Backlogs":
                z = -z
            impact = imp * z
        else:
            impact = imp * (0.5 if profile[feat] in ["Yes"] else 0.0)
        rows.append({"Feature": feat, "Impact": impact})
    expl_df = pd.DataFrame(rows)
    expl_df["AbsImpact"] = expl_df["Impact"].abs()
    expl_df = expl_df.sort_values("AbsImpact", ascending=False).head(top_n)
    expl_df["Direction"] = np.where(expl_df["Impact"] > 0, "Increases", "Decreases")
    return expl_df[["Feature", "Impact", "Direction"]], "Feature-importance heuristic (install `shap` for exact attribution)"


def compute_role_match(profile: dict):
    """Compute a 0-100 match score for every job role + skill gaps."""
    results = []
    for role, spec in JOB_ROLES.items():
        weights = spec["weights"]
        total, achieved = 0.0, 0.0
        gaps = []
        for feat, target in weights.items():
            if feat == "Backlogs_Inverse":
                actual = max(0, 10 - profile["Backlogs"] * 2.5)
                target_val = target
            else:
                actual = profile.get(feat, 0)
                target_val = target
            total += target_val
            ratio = min(actual / target_val, 1.0) if target_val > 0 else 1.0
            achieved += ratio * target_val
            if actual < target_val:
                gaps.append({
                    "skill": feat.replace("_", " "),
                    "current": round(actual, 1),
                    "required": target_val,
                    "gap": round(target_val - actual, 1),
                })
        match_pct = round((achieved / total) * 100, 1) if total > 0 else 0
        gaps = sorted(gaps, key=lambda g: -g["gap"])
        results.append({
            "role": role, "icon": spec["icon"], "match": match_pct,
            "gaps": gaps, "core_skills": spec["core_skills"], "description": spec["description"],
        })
    return sorted(results, key=lambda r: -r["match"])


def extract_resume_text(uploaded_file):
    """Extract raw text from an uploaded PDF or TXT resume."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        import pdfplumber
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def analyze_resume(text: str):
    """Keyword-based resume skill extraction + category coverage score."""
    text_lower = text.lower()
    found = {}
    for category, keywords in RESUME_SKILL_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text_lower]
        found[category] = matched
    total_possible = sum(len(v) for v in RESUME_SKILL_KEYWORDS.values())
    total_found = sum(len(v) for v in found.values())
    coverage = round((total_found / total_possible) * 100, 1) if total_possible else 0
    word_count = len(text.split())
    return found, coverage, word_count


def generate_improvement_plan(profile: dict, probability: float, gaps: list):
    """Produce a prioritized, personalized improvement plan."""
    plan = []
    if profile["CGPA"] < 7.0:
        plan.append(("Academics", "Raise CGPA above 7.0", "Aim for consistent 8+ in upcoming semesters; "
                                                             "target weak subjects with weekly revision.", "High"))
    if profile["Backlogs"] > 0:
        plan.append(("Academics", "Clear pending backlogs", "Backlogs strongly reduce placement odds — "
                                                              "prioritize clearing them in the next attempt.", "Critical"))
    if profile["Internships"] < 2:
        plan.append(("Experience", "Complete at least 2 internships", "Apply to 2-3 month internships "
                                                                        "(virtual internships on Internshala/AICTE count too).", "High"))
    if profile["Projects"] < 3:
        plan.append(("Experience", "Build 3+ solid projects", "Pick projects that show end-to-end skills "
                                                                "(1 full-stack/ML project + 2 smaller ones), and host them on GitHub.", "High"))
    if profile["Certifications"] < 2:
        plan.append(("Skills", "Earn 2+ relevant certifications", "NPTEL / Coursera / Udemy certifications "
                                                                    "in your target domain build credibility fast.", "Medium"))
    if profile["Aptitude_Score"] < 65:
        plan.append(("Aptitude", "Improve quantitative & logical aptitude", "Practice 30 mins/day on "
                                                                             "IndiaBix/PrepInsta; most core companies screen with aptitude tests.", "High"))
    if profile["Technical_Skill"] < 6.5 or profile["Coding_Skill"] < 6.5:
        plan.append(("Skills", "Strengthen coding & technical fundamentals", "Solve 3 DSA problems/day on "
                                                                              "LeetCode/HackerRank; revise core CS subjects.", "Critical"))
    if profile["Communication_Skill"] < 6.0:
        plan.append(("Soft Skills", "Improve communication skills", "Join a speaking club, do 2 mock interviews "
                                                                      "a week, and practice structured self-introduction.", "Medium"))
    if profile["Placement_Training"] == "No":
        plan.append(("Preparation", "Enroll in placement training", "Structured training covering resume, "
                                                                      "GD, and interviews meaningfully lifts placement odds.", "Medium"))
    if profile["LinkedIn_GitHub_Activity"] < 5:
        plan.append(("Visibility", "Build your LinkedIn & GitHub presence", "Recruiters screen profiles — "
                                                                             "post projects, keep GitHub active, and network on LinkedIn.", "Low"))

    if gaps:
        top_gap = gaps[0]
        plan.append(("Role Fit", f"Close gap in {top_gap['skill']}", f"You're {top_gap['gap']} points below the "
                                                                       f"target for your best-matching role — focused practice here has outsized impact.", "High"))

    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    plan = sorted(plan, key=lambda x: priority_order.get(x[3], 4))
    if not plan:
        plan.append(("Overall", "Maintain your strong profile", "You're in great shape — keep practicing "
                                                                  "mock interviews and stay updated with industry trends.", "Low"))
    return plan


def probability_badge(prob):
    if prob >= 0.70:
        return '<span class="badge badge-green">High Chance</span>'
    elif prob >= 0.40:
        return '<span class="badge badge-yellow">Moderate Chance</span>'
    else:
        return '<span class="badge badge-red">Needs Improvement</span>'


def load_progress_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "student_name", "probability", "employability_score", "package"])


def save_progress_entry(name, probability, emp_score, package):
    log = load_progress_log()
    new_row = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_name": name,
        "probability": round(probability * 100, 2),
        "employability_score": round(emp_score, 2),
        "package": round(package, 2),
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return log


# ========================================================================================
# COMPANY-SPECIFIC READINESS SCORE
# ========================================================================================
COMPANY_CRITERIA = {
    "TCS (NQT)":       {"icon": "🔵", "tier": "Mass Recruiter",  "cgpa": 6.0, "max_backlogs": 1, "aptitude": 50, "technical": 5.0, "coding": 4.5, "communication": 5.0},
    "Infosys":         {"icon": "🔷", "tier": "Mass Recruiter",  "cgpa": 6.5, "max_backlogs": 1, "aptitude": 55, "technical": 5.0, "coding": 5.0, "communication": 5.5},
    "Wipro":           {"icon": "🟣", "tier": "Mass Recruiter",  "cgpa": 6.0, "max_backlogs": 1, "aptitude": 50, "technical": 5.0, "coding": 4.5, "communication": 5.0},
    "Accenture":       {"icon": "🟪", "tier": "Mass Recruiter",  "cgpa": 6.5, "max_backlogs": 0, "aptitude": 55, "technical": 5.5, "coding": 5.0, "communication": 6.0},
    "Cognizant":       {"icon": "🔶", "tier": "Mass Recruiter",  "cgpa": 6.0, "max_backlogs": 1, "aptitude": 50, "technical": 5.0, "coding": 5.0, "communication": 5.0},
    "Deloitte (Consulting)": {"icon": "🟢", "tier": "Consulting", "cgpa": 6.5, "max_backlogs": 0, "aptitude": 60, "technical": 5.5, "coding": 4.5, "communication": 7.0},
    "Amazon":          {"icon": "🟠", "tier": "Product-Based",   "cgpa": 7.0, "max_backlogs": 0, "aptitude": 70, "technical": 7.5, "coding": 7.5, "communication": 6.0},
    "Microsoft":       {"icon": "🪟", "tier": "Product-Based",   "cgpa": 7.5, "max_backlogs": 0, "aptitude": 75, "technical": 8.0, "coding": 8.0, "communication": 6.5},
    "Google":          {"icon": "🌈", "tier": "Product-Based",   "cgpa": 8.0, "max_backlogs": 0, "aptitude": 80, "technical": 9.0, "coding": 9.0, "communication": 7.0},
}


def compute_company_readiness(profile: dict):
    """For each named company, compute a 0-100 readiness % against realistic public hiring bars,
    plus the specific gaps a student needs to close."""
    results = []
    for name, c in COMPANY_CRITERIA.items():
        ratios = []
        gaps = []

        cgpa_ratio = min(profile["CGPA"] / c["cgpa"], 1.0) if c["cgpa"] > 0 else 1.0
        ratios.append(cgpa_ratio)
        if profile["CGPA"] < c["cgpa"]:
            gaps.append(f"Raise CGPA to {c['cgpa']} (currently {profile['CGPA']})")

        if profile["Backlogs"] <= c["max_backlogs"]:
            back_ratio = 1.0
        else:
            back_ratio = max(0.0, c["max_backlogs"] / profile["Backlogs"]) if profile["Backlogs"] > 0 else 1.0
            gaps.append(f"Clear backlogs down to {c['max_backlogs']} (currently {profile['Backlogs']})")
        ratios.append(back_ratio)

        apt_ratio = min(profile["Aptitude_Score"] / c["aptitude"], 1.0) if c["aptitude"] > 0 else 1.0
        ratios.append(apt_ratio)
        if profile["Aptitude_Score"] < c["aptitude"]:
            gaps.append(f"Raise Aptitude Score to {c['aptitude']} (currently {profile['Aptitude_Score']})")

        tech_ratio = min(profile["Technical_Skill"] / c["technical"], 1.0) if c["technical"] > 0 else 1.0
        ratios.append(tech_ratio)
        if profile["Technical_Skill"] < c["technical"]:
            gaps.append(f"Raise Technical Skill to {c['technical']} (currently {profile['Technical_Skill']})")

        code_ratio = min(profile["Coding_Skill"] / c["coding"], 1.0) if c["coding"] > 0 else 1.0
        ratios.append(code_ratio)
        if profile["Coding_Skill"] < c["coding"]:
            gaps.append(f"Raise Coding Skill to {c['coding']} (currently {profile['Coding_Skill']})")

        comm_ratio = min(profile["Communication_Skill"] / c["communication"], 1.0) if c["communication"] > 0 else 1.0
        ratios.append(comm_ratio)
        if profile["Communication_Skill"] < c["communication"]:
            gaps.append(f"Raise Communication Skill to {c['communication']} (currently {profile['Communication_Skill']})")

        readiness_pct = round((sum(ratios) / len(ratios)) * 100, 1)
        if readiness_pct >= 85:
            status_label, badge_class = "Ready to Apply", "badge-green"
        elif readiness_pct >= 60:
            status_label, badge_class = "Almost Ready", "badge-yellow"
        else:
            status_label, badge_class = "Needs Preparation", "badge-red"

        results.append({
            "name": name, "icon": c["icon"], "tier": c["tier"],
            "readiness_pct": readiness_pct, "status_label": status_label,
            "badge_class": badge_class, "gaps": gaps,
        })
    return sorted(results, key=lambda r: -r["readiness_pct"])


# ========================================================================================
# 🤖 ASK YOUR MENTOR — rebuilt as an intent-scoring engine (not a fragile if/elif chain)
# Every question is scored against ~25 intents by keyword overlap; the best match wins,
# so the mentor now handles far more phrasings, general "what is X" questions, comparisons,
# and encouragement — not just a handful of exact trigger phrases.
# ========================================================================================
DEFINITIONS_BANK = {
    "aptitude": "an aptitude test measures quantitative, logical reasoning, and verbal ability — it's usually the "
                "first screening round at most mass-recruiter and product companies.",
    "ats": "an ATS (Applicant Tracking System) is software recruiters use to scan resumes for keywords before a "
           "human ever sees them — which is why keyword coverage on your resume matters.",
    "dsa": "DSA (Data Structures & Algorithms) is the core subject tested in technical interviews — arrays, "
           "linked lists, trees, graphs, sorting/searching, and time/space complexity.",
    "lpa": "LPA means 'Lakhs Per Annum' — it's how annual salary packages are usually quoted in India (e.g. 6 LPA = ₹6,00,000/year).",
    "ctc": "CTC (Cost to Company) is the full annual package a company reports, including base salary, bonuses, and benefits — "
           "your in-hand salary is usually lower than the CTC figure.",
    "backlog": "a 'backlog' is a subject/exam you haven't cleared yet — many companies set a strict 'no active backlog' "
               "cutoff at the time of hiring, so clearing them early matters a lot.",
    "off-campus": "off-campus hiring means applying directly to a company's careers page or via referrals, rather than "
                  "through your college's placement cell — common for product-based companies like Google/Amazon.",
    "on-campus": "on-campus hiring is when companies visit your college directly through the placement/training cell.",
    "gd": "a GD (Group Discussion) round tests communication, confidence, and how you engage with a group on a given topic "
          "— common in service-based company hiring processes.",
    "notice period": "a notice period is the time you must serve at a current job before leaving — not usually relevant "
                      "for a first placement, but it comes up once you're already employed.",
}

MENTOR_TOPIC_SUGGESTIONS = [
    "\"Am I ready for placements?\"", "\"What should I improve first?\"", "\"What are my strengths?\"",
    "\"Which companies fit me?\"", "\"Which job role suits me?\"", "\"How's my resume?\"",
    "\"What package can I expect?\"", "\"What's my roadmap for the next 3 months?\"",
    "\"What is DSA / ATS / LPA?\"", "\"Tips for interviews?\"",
]


def _mentor_greeting(student_name):
    return (f"Hi **{student_name}** 👋 — I'm your AI Career Mentor. Ask me anything about your placement "
            f"chances, strengths and gaps, which companies or roles fit you, your resume, expected package, "
            f"or your prep roadmap. You can type a question below or tap a quick question to get started.")


def _build_mentor_context(profile):
    label, probability, package = predict_profile(profile)
    emp_score = compute_employability_score(profile)
    role_matches = compute_role_match(profile)
    best_role = role_matches[0]
    plan = generate_improvement_plan(profile, probability, best_role["gaps"])
    company_results = compute_company_readiness(profile)
    top_company = company_results[0]
    return {
        "label": label, "probability": probability, "package": package, "emp_score": emp_score,
        "role_matches": role_matches, "best_role": best_role, "plan": plan,
        "company_results": company_results, "top_company": top_company,
    }


def generate_mentor_reply(query: str, profile: dict, student_name: str) -> str:
    """Intent-scored, profile-aware mentor response engine (fully offline, deterministic)."""
    q = query.lower().strip()
    ctx = _build_mentor_context(profile)
    probability, package, emp_score = ctx["probability"], ctx["package"], ctx["emp_score"]
    role_matches, best_role, plan = ctx["role_matches"], ctx["best_role"], ctx["plan"]
    company_results, top_company = ctx["company_results"], ctx["top_company"]

    def greet_prefix():
        return f"**{student_name}**, "

    # ---- intent handlers (each returns the response string) ----
    def h_greeting():
        return _mentor_greeting(student_name)

    def h_thanks():
        return f"You're welcome, {student_name}! Keep working through your Improvement Plan — I'm here whenever you need guidance. 🎯"

    def h_readiness():
        return (f"{greet_prefix()}your current **placement probability is {probability*100:.1f}%** "
                f"({'High Chance' if probability>=0.7 else 'Moderate Chance' if probability>=0.4 else 'Needs Improvement'}), "
                f"with an employability score of **{emp_score:.1f}/100**. "
                f"{'You are in a strong position — keep sharpening interview skills in the Interview Lab.' if probability>=0.7 else 'A few focused improvements can meaningfully raise this — ask me what to improve first!'}")

    def h_emp_score():
        return (f"{greet_prefix()}your **employability score is {emp_score:.1f}/100** — it's a transparent blend of "
                f"academics, experience, aptitude, and skills (separate from the ML placement probability, which is "
                f"{probability*100:.1f}%). Raising CGPA, clearing backlogs, and adding internships/projects move this "
                f"score the most.")

    def h_improve():
        top_items = plan[:3]
        bullets = "\n".join([f"- **{t[1]}** — {t[2]}" for t in top_items])
        return f"{greet_prefix()}here are your top priorities right now:\n\n{bullets}"

    def h_weaknesses():
        low_items = [t for t in plan if t[3] in ("Critical", "High")][:3] or plan[:2]
        bullets = "\n".join([f"- **{t[1]}**" for t in low_items])
        return f"{greet_prefix()}your biggest weak spots right now are:\n\n{bullets}\n\nAsk me *\"how do I improve\"* for the fix for each."

    def h_strengths():
        strong_points = []
        if profile["CGPA"] >= 7.5: strong_points.append("a strong CGPA")
        if profile["Backlogs"] == 0: strong_points.append("zero backlogs")
        if profile["Internships"] >= 2: strong_points.append("solid internship experience")
        if profile["Projects"] >= 3: strong_points.append("a good project portfolio")
        if profile["Technical_Skill"] >= 7: strong_points.append("strong technical skill")
        if profile["Coding_Skill"] >= 7: strong_points.append("strong coding ability")
        if profile["Communication_Skill"] >= 7: strong_points.append("good communication skill")
        if profile["Aptitude_Score"] >= 70: strong_points.append("a high aptitude score")
        if not strong_points:
            return (f"{greet_prefix()}your profile doesn't have a stand-out strength yet on paper — but that's exactly "
                     f"what the Improvement Plan is for. Ask me *\"what should I improve\"* to find your fastest wins.")
        return f"{greet_prefix()}your current strengths are: **{', '.join(strong_points)}**. Lean on these in interviews and your resume summary."

    def h_role():
        top3 = role_matches[:3]
        bullets = "\n".join([f"- {r['icon']} **{r['role']}** — {r['match']}% match" for r in top3])
        return (f"{greet_prefix()}your best-fit role right now is **{best_role['icon']} {best_role['role']}** "
                f"at **{best_role['match']}%** match. Your top 3 matches:\n\n{bullets}\n\nSee the **Job Role Matching** "
                f"section of your report for the full skill-gap breakdown.")

    def h_company():
        return (f"{greet_prefix()}based on your current profile, **{top_company['icon']} {top_company['name']}** "
                f"is your best-fit company target at **{top_company['readiness_pct']}% readiness** "
                f"({top_company['status_label']}). Check the **Company Readiness** page in the sidebar for "
                f"a full breakdown across all 9 companies and exactly what to close for each.")

    def h_backlog():
        if profile["Backlogs"] > 0:
            return (f"{greet_prefix()}you currently have **{profile['Backlogs']} active backlog(s)**. This is one "
                     f"of the strongest negative factors in placement prediction — clearing even one backlog can "
                     f"noticeably raise your probability. Prioritize this above almost everything else.")
        return f"{greet_prefix()}good news — you have **zero backlogs**. That's a strong positive signal for recruiters."

    def h_resume():
        return (f"{greet_prefix()}head to the **Resume Analysis** section of your full report — upload your resume "
                f"there and I'll extract detected skills, keyword coverage %, and word count automatically, then "
                f"you can pull those skills straight into your profile.")

    def h_interview():
        mi = profile.get("Mock_Interview_Score", 0)
        tip = "solid — keep practicing to stay sharp" if mi >= 7 else "an area to actively work on"
        return (f"{greet_prefix()}your logged mock interview score is **{mi}/10**, which is {tip}. Head to the "
                f"**🎤 Interview Lab** in the sidebar — practice HR, technical, aptitude, or role-specific questions "
                f"and get instant, structured feedback on every answer.")

    def h_cgpa():
        return (f"{greet_prefix()}your CGPA is **{profile['CGPA']}/10**. "
                f"{'This is comfortably above the common recruiter cutoff of 7.0.' if profile['CGPA']>=7.0 else 'Most recruiters set cutoffs around 6.0–7.0 — raising this even slightly widens your eligible companies significantly.'}")

    def h_package():
        label = ctx["label"]
        if label and package > 0:
            return (f"{greet_prefix()}based on your current profile, the model estimates a predicted package of "
                     f"around **₹{package:.2f} LPA**. This scales up with CGPA, internships, projects, and technical/coding skill "
                     f"— raising the ones flagged in your Improvement Plan will lift this estimate too.")
        return (f"{greet_prefix()}your placement probability is currently below the threshold where a reliable package "
                f"estimate kicks in. Focus on your top improvement priorities first — once your placement probability "
                f"rises, check back here for a package estimate.")

    def h_timeline():
        crit = [t for t in plan if t[3] == "Critical"]
        high = [t for t in plan if t[3] == "High"]
        return (f"{greet_prefix()}a realistic focus order: **this week** — {crit[0][1] if crit else (high[0][1] if high else 'polish your resume')}; "
                f"**this month** — work through the rest of your High-priority items in the Improvement Plan; "
                f"**ongoing** — DSA practice, mock interviews in the Interview Lab, and keeping LinkedIn/GitHub active. "
                f"See the **30-day action sprint** at the bottom of the Improvement Plan section for a week-by-week layout.")

    def h_motivation():
        return (f"{greet_prefix()}feeling stretched about placements is completely normal — almost every student goes "
                f"through it. The good news is your prep is a series of small, concrete steps, not one big leap. "
                f"Pick just **one** item from your Improvement Plan to focus on this week, and use the **Interview Lab** "
                f"for low-pressure practice. Progress compounds faster than it feels like it will.")

    def h_comparison_company_type():
        return ("**Service-based companies** (TCS, Infosys, Wipro, Cognizant, Accenture) hire in bulk, screen mainly on "
                "CGPA cutoff + aptitude test + basic coding/communication, and are a great first target with lower "
                "entry bars. **Product-based companies** (Google, Microsoft, Amazon) hire fewer people but pay far more, "
                "and screen heavily on DSA depth, coding rounds, and project quality. Check the **Company Readiness** "
                "page to see exactly where you stand against both types.")

    def h_certifications():
        return (f"{greet_prefix()}you currently have **{profile['Certifications']} certification(s)** logged. "
                f"{'Two or more relevant certifications (NPTEL/Coursera/Udemy) in your target domain meaningfully build recruiter credibility.' if profile['Certifications'] < 2 else 'That is a solid number — make sure they are listed prominently on your resume.'}")

    def h_projects():
        return (f"{greet_prefix()}you currently have **{profile['Projects']} project(s)** logged. Aim for **3+**: "
                f"one full end-to-end project (ideally full-stack or ML) plus two smaller focused ones, all hosted "
                f"on GitHub with a clean README — this is one of the highest-leverage things recruiters actually look at.")

    def h_internship():
        return (f"{greet_prefix()}you currently have **{profile['Internships']} internship(s)** logged. "
                f"{'Aim for at least 2 — even a 2-3 month virtual internship via Internshala/AICTE counts and meaningfully improves your odds.' if profile['Internships'] < 2 else 'That is a solid base — make sure your resume clearly states the impact/outcome of each internship.'}")

    def h_linkedin_github():
        return (f"{greet_prefix()}your LinkedIn/GitHub activity score is **{profile['LinkedIn_GitHub_Activity']}/10**. "
                f"Recruiters do screen public profiles — keep your GitHub active with your projects, and post about "
                f"what you're building/learning on LinkedIn at least a couple of times a month.")

    def h_communication():
        return (f"{greet_prefix()}your communication skill is **{profile['Communication_Skill']}/10**. "
                f"{'This is a genuine strength — use it well in interviews and group discussions.' if profile['Communication_Skill']>=7 else 'Practicing structured self-introductions and 2 mock interviews a week in the Interview Lab is the fastest way to move this.'}")

    def h_help():
        return ("I can help with: your **placement readiness**, **strengths & weaknesses**, best-fit **role** and "
                "**company**, an **expected package** estimate, a **prep roadmap/timeline**, your **resume**, and "
                "general placement terms (ask *\"what is DSA\"* or *\"what is ATS\"*, for example). For hands-on "
                "practice, open the **🎤 Interview Lab** in the sidebar.")

    def h_definitions():
        for term, meaning in DEFINITIONS_BANK.items():
            if term in q:
                return f"{meaning[0].upper() + meaning[1:]}"
        return None

    # (name, keyword list, handler) — order matters only as a tie-breaker
    INTENTS = [
        ("greeting", ["hi", "hello", "hey", "namaste", "hai", "good morning", "good evening"], h_greeting),
        ("thanks", ["thank", "thanks", "bye", "goodbye", "see you"], h_thanks),
        ("readiness", ["ready", "chance", "probability", "placed", "placement odds", "will i get", "odds of", "am i ready"], h_readiness),
        ("emp_score", ["employability score", "what does my score mean", "what is my score"], h_emp_score),
        ("weaknesses", ["weakness", "weak point", "weak area", "what's wrong", "holding me back", "bad at"], h_weaknesses),
        ("improve", ["improve", "better", "increase", "boost", "what should i", "how do i get", "priorities"], h_improve),
        ("strengths", ["strength", "strong point", "good at", "doing well", "what am i good at"], h_strengths),
        ("role", ["role", "job profile", "career path", "which job", "suited", "suitable job", "which position"], h_role),
        ("company", ["company", "companies", "amazon", "tcs", "infosys", "wipro", "google", "microsoft", "target company", "accenture", "cognizant", "deloitte"], h_company),
        ("backlog", ["backlog"], h_backlog),
        ("resume", ["resume", "cv"], h_resume),
        ("interview", ["interview", "mock", "hr round", "technical round"], h_interview),
        ("cgpa", ["cgpa", "gpa", "marks", "academic percentage"], h_cgpa),
        ("package", ["package", "salary", "lpa", "ctc", "pay", "compensation"], h_package),
        ("timeline", ["how long", "timeline", "roadmap", "how many months", "when should i", "plan for next", "schedule"], h_timeline),
        ("motivation", ["stressed", "demotivated", "anxious", "nervous", "worried", "scared", "overwhelmed", "hopeless", "give up"], h_motivation),
        ("comparison", ["difference between", "service based", "product based", "vs product", "mass recruiter vs"], h_comparison_company_type),
        ("certifications", ["certification", "certificate", "course completed", "nptel", "coursera"], h_certifications),
        ("projects", ["project"], h_projects),
        ("internship", ["internship", "intern"], h_internship),
        ("linkedin_github", ["linkedin", "github", "git hub", "portfolio"], h_linkedin_github),
        ("communication", ["communication", "soft skill", "speaking", "presentation skill"], h_communication),
        ("help", ["what can you do", "help me", "how does this work", "what is this platform", "features"], h_help),
    ]

    # definitions get first shot for explicit "what is X" phrasing so profile-specific
    # intents (e.g. "backlog") don't swallow a pure definitional question
    if q.startswith("what is") or q.startswith("what's") or "meaning of" in q or "define" in q:
        d = h_definitions()
        if d:
            return d

    best_intent, best_score = None, 0
    for name, keywords, handler in INTENTS:
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score, best_intent = score, handler

    if best_intent is not None:
        return best_intent()

    d = h_definitions()
    if d:
        return d

    # Fallback — general, still profile-aware, with concrete suggestions
    top_item = plan[0]
    suggestions = ", ".join(random.sample(MENTOR_TOPIC_SUGGESTIONS, k=4))
    return (f"{greet_prefix()}here's a quick snapshot: **{probability*100:.1f}%** placement probability, "
            f"best-fit role **{best_role['role']}** ({best_role['match']}% match), and your top priority right now is "
            f"**{top_item[1]}**. I might not have caught your exact question — try asking things like {suggestions}.")


# ========================================================================================
# 🎤 INTERVIEW LAB — mock interview practice with instant, structured feedback
# (replaces the old Goal Optimizer page)
# ========================================================================================
HR_BEHAVIORAL_QUESTIONS = [
    "Tell me about yourself.",
    "Tell me about a time you faced a conflict in a team and how you handled it.",
    "Describe a situation where you failed at something. What did you learn?",
    "Tell me about a time you had to meet a tight deadline.",
    "Describe a time you had to learn a new skill quickly.",
    "Tell me about a time you disagreed with a teammate or professor. What happened?",
    "Why should we hire you over other candidates?",
    "Where do you see yourself in 5 years?",
    "Describe a project you're most proud of and why.",
    "Tell me about a time you took initiative without being asked.",
]

TECHNICAL_FUNDAMENTAL_QUESTIONS = [
    "Explain the difference between an array and a linked list.",
    "What is the time complexity of binary search, and why?",
    "Explain the difference between a stack and a queue with an example.",
    "What is normalization in DBMS, and why is it used?",
    "Explain the difference between process and thread in an operating system.",
    "What is the difference between TCP and UDP?",
    "Explain OOP concepts: encapsulation, inheritance, polymorphism, abstraction.",
    "What is a hash table, and how does it achieve O(1) lookup on average?",
    "Explain the difference between SQL and NoSQL databases.",
    "What is recursion, and what's a risk of using it carelessly?",
]

APTITUDE_REASONING_QUESTIONS = [
    "If a train travels 300 km in 4 hours, what's its average speed, and how did you calculate it?",
    "A shopkeeper marks up an item by 25% and then gives a 10% discount — explain how to find the final profit percentage.",
    "Explain how you would approach a puzzle involving seating arrangements in a circle.",
    "If the ratio of ages of A and B is 3:5 and their sum is 40, explain how you'd find each age.",
    "Explain your approach to a simple probability question, like the chance of drawing 2 aces from a deck.",
    "Describe how you'd solve a work-and-time problem where two people work at different rates.",
    "Explain how you would identify the next number in a logical number series.",
    "How would you approach a data interpretation question given a bar chart of sales figures?",
]

STAR_KEYWORDS = {
    "situation": ["situation", "context", "at the time", "during", "when i was", "background"],
    "task": ["task", "goal", "needed to", "had to", "objective", "responsible for"],
    "action": ["i did", "i decided", "action", "approach", "so i", "i started", "i built", "i led", "i implemented"],
    "result": ["result", "outcome", "as a result", "in the end", "this led to", "improved", "achieved", "successfully"],
}

TECH_DEPTH_MARKERS = [
    "because", "for example", "time complexity", "space complexity", "algorithm", "data structure",
    "trade-off", "tradeoff", "in practice", "compared to", "average case", "worst case", "o(", "example:",
]


def evaluate_answer(category: str, answer: str):
    """Rule-based, offline scoring of an interview answer. Returns (score_out_of_10, tips list)."""
    text = answer.strip()
    tips = []
    if not text:
        return 0, ["You didn't enter an answer yet — try to speak/write it out loud, even roughly, before checking feedback."]

    words = text.split()
    word_count = len(words)
    lower = text.lower()

    # --- length component (0-4) ---
    if word_count < 12:
        length_score = 1
        tips.append("Your answer is quite short — aim for at least 6-8 sentences so you show depth, not just a one-liner.")
    elif word_count < 40:
        length_score = 2.5
        tips.append("Good start — try adding one more concrete detail or example to make the answer fuller.")
    elif word_count <= 150:
        length_score = 4
    else:
        length_score = 3
        tips.append("This answer is quite long — in a real interview, aim to be a bit more concise and focused.")

    # --- structure component (0-4) ---
    structure_score = 0
    if category == "HR / Behavioral":
        hits = [comp for comp, kws in STAR_KEYWORDS.items() if any(k in lower for k in kws)]
        structure_score = min(len(hits), 4)
        missing = [c for c in STAR_KEYWORDS if c not in hits]
        if missing:
            tips.append(f"Try to clearly cover the **STAR** structure — you're missing: {', '.join(m.title() for m in missing)}.")
        else:
            tips.append("Nice — your answer already touches all four STAR components (Situation, Task, Action, Result).")
    else:
        depth_hits = sum(1 for m in TECH_DEPTH_MARKERS if m in lower)
        structure_score = min(depth_hits * 1.3, 4)
        if depth_hits == 0:
            tips.append("Add reasoning, not just the final answer — mention *why*, an example, or a trade-off/complexity to show depth.")
        else:
            tips.append("Good — you're explaining your reasoning, not just stating a conclusion.")

    # --- confidence/clarity component (0-2) ---
    filler_words = ["um", "like", "basically", "actually", "kind of", "sort of", "i think maybe", "not sure"]
    filler_hits = sum(lower.count(f) for f in filler_words)
    clarity_score = 2 if filler_hits == 0 else (1 if filler_hits <= 2 else 0.5)
    if filler_hits > 2:
        tips.append("Watch filler phrases like 'kind of' / 'not sure' / 'basically' — stating things directly reads as more confident.")

    total = round(min(length_score + structure_score + clarity_score, 10), 1)
    if total >= 8:
        tips.insert(0, "Strong answer overall.")
    elif total >= 5.5:
        tips.insert(0, "Solid attempt — a few tweaks will make this noticeably stronger.")
    else:
        tips.insert(0, "This needs more work before an interview — use the tips below to rebuild it.")

    return total, tips


def _role_specific_questions(best_role):
    core_skills = best_role.get("core_skills", [])[:6]
    qs = [f"Describe a project or situation where you applied {skill}." for skill in core_skills]
    qs.append(f"Why do you think you're a good fit for a {best_role['role']} role?")
    qs.append(f"What would you want to learn first if you started as a {best_role['role']} tomorrow?")
    return qs if qs else ["Tell me why this role interests you and what you've done to prepare for it."]


# --------------------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------------------------------------------
st.session_state.setdefault("assessment_run", False)
st.session_state.setdefault("profile", None)
st.session_state.setdefault("student_name", "Student")
st.session_state.setdefault("whatif_profile", None)
st.session_state.setdefault("app_page", "📝 Student Profile")
st.session_state.setdefault("pending_page", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("pending_chat_msg", None)
st.session_state.setdefault("interview_category", None)
st.session_state.setdefault("interview_question", None)
st.session_state.setdefault("interview_log", [])

# --------------------------------------------------------------------------------------
# SIDEBAR — NAVIGATION ONLY (Student Profile is now a main-screen page)
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 🎓 AI Employability Platform")
if st.session_state["assessment_run"]:
    st.sidebar.success(f"✅ Assessment ready for **{st.session_state['student_name']}**")
else:
    st.sidebar.warning("👋 Start with **Student Profile**, then run your assessment.")

st.sidebar.markdown("### 🧭 Navigate")
NAV_OPTIONS = [
    "📝 Student Profile",
    "🏠 Home / Full Report",
]
st.sidebar.markdown("###### ✨ Explore More")
NEW_NAV_OPTIONS = [
    "🤖 Ask Your Mentor",
    "🎤 Interview Lab",
    "🏢 Company Readiness",
]
ALL_NAV_OPTIONS = NAV_OPTIONS + NEW_NAV_OPTIONS

if st.session_state.get("pending_page") is not None:
    st.session_state["app_page"] = st.session_state["pending_page"]
    st.session_state["pending_page"] = None

nav = st.sidebar.radio(
    "Explore",
    ALL_NAV_OPTIONS,
    key="app_page",
    label_visibility="collapsed",
)

# --------------------------------------------------------------------------------------
# HEADER + HOW IT WORKS
# --------------------------------------------------------------------------------------
HIW_STEPS = [
    ("👤", "Student Profile", "Fill in your details on the main screen."),
    ("🧠", "AI Assessment", "Click Run — get your placement probability instantly."),
    ("📊", "Full Report", "Every insight renders automatically, one after another."),
    ("🤖", "Ask Your Mentor", "Chat with your AI mentor for personalized guidance."),
    ("🎤", "Practice Interviews", "Answer mock questions and get instant, structured feedback."),
]


def render_how_it_works(steps):
    html = ['<div class="hiw-wrap">']
    html.append('<div class="hiw-title">How It Works</div>')
    html.append('<div class="hiw-sub">From your profile to a placement-ready action plan, in one click</div>')
    html.append('<div class="hiw-row">')
    for i, (icon, title, desc) in enumerate(steps, start=1):
        html.append(
            f'<div class="hiw-card">'
            f'<div class="hiw-num">{i:02d}</div>'
            f'<div class="hiw-card-title">{icon} {title}</div>'
            f'<div class="hiw-card-desc">{desc}</div>'
            f'</div>'
        )
        if i < len(steps):
            html.append('<div class="hiw-arrow">&#8594;</div>')
    html.append('</div></div>')
    return "".join(html)


def render_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>🎓 {title}</h1>
            <p>{subtitle}</p>
            <div class="hero-icons">🧠 📊 🤖 🎤 🏢</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(subtitle):
    render_hero("AI Employability & Career Intelligence Platform", subtitle)
    st.markdown(render_how_it_works(HIW_STEPS), unsafe_allow_html=True)
    st.markdown("---")


# ========================================================================================
# STUDENT PROFILE — full main-screen page
# ========================================================================================
def render_profile_page():
    render_header(
        "Fill in your details below, then click <b>Run Employability Assessment</b> "
        "to unlock your full career intelligence report and AI-powered tools."
    )

    st.markdown(
        """
        <div class="metric-card" style="display:flex; align-items:center; gap:18px; margin-bottom:22px;
             background:linear-gradient(135deg, rgba(139,92,246,0.14), rgba(34,211,238,0.08));">
            <div style="font-size:2.6rem; line-height:1;">🪪</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:1.15rem;">
                    Build your Career Profile
                </div>
                <div style="color:var(--ink-dim); font-size:0.92rem;">
                    Four quick sections — Basics, Academics, Experience, and Skills & Aptitude.
                    Everything you enter feeds your placement prediction, your AI mentor, and the Interview Lab.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">📝 Student Profile</div>', unsafe_allow_html=True)

    with st.form("profile_form"):
        st.markdown("##### 👤 Basics")
        top1, top2, top3 = st.columns(3)
        with top1:
            name = st.text_input("Student Name", value=st.session_state.get("student_name", "Student"))
        with top2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with top3:
            branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"])

        with st.expander("🎓  Academics", expanded=True):
            st.caption("CGPA and backlogs are the single strongest signals in the placement model.")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                cgpa = st.slider("CGPA (0-10)", 4.0, 10.0, 7.2, 0.05)
            with a2:
                ssc = st.slider("SSC / 10th Marks (%)", 40, 100, 78)
            with a3:
                hsc = st.slider("HSC / 12th Marks (%)", 40, 100, 75)
            with a4:
                backlogs = st.number_input("Active Backlogs", 0, 10, 0)

        with st.expander("💼  Experience", expanded=True):
            st.caption("Internships and projects carry the most weight after academics.")
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                internships = st.number_input("Internships Completed", 0, 10, 1)
            with e2:
                projects = st.number_input("Projects Built", 0, 15, 2)
            with e3:
                certifications = st.number_input("Certifications Earned", 0, 15, 1)
            with e4:
                workshops = st.number_input("Workshops / Bootcamps Attended", 0, 10, 1)

        with st.expander("🧠  Skills & Aptitude", expanded=True):
            st.caption("Be honest here — these numbers drive your role match, skill-gap radar, and mentor advice.")
            s1, s2, s3 = st.columns(3)
            with s1:
                aptitude = st.slider("Aptitude Test Score (0-100)", 0, 100, 60)
                technical = st.slider("Technical Skill", 0.0, 10.0, 6.0, 0.1)
                coding = st.slider("Coding Skill", 0.0, 10.0, 6.0, 0.1)
            with s2:
                communication = st.slider("Communication Skill", 0.0, 10.0, 6.0, 0.1)
                soft = st.slider("Soft Skill (teamwork, adaptability)", 0.0, 10.0, 6.0, 0.1)
                extracurricular = st.slider("Extracurricular Involvement", 0.0, 10.0, 5.0, 0.1)
            with s3:
                leadership = st.slider("Leadership Experience", 0.0, 10.0, 5.0, 0.1)
                mock_interview = st.slider("Mock Interview Score", 0.0, 10.0, 5.5, 0.1)
                linkedin_github = st.slider("LinkedIn/GitHub Activity", 0.0, 10.0, 5.0, 0.1)

            placement_training = st.selectbox("Enrolled in Placement Training?", ["Yes", "No"])

        submitted = st.form_submit_button("🚀 Run Employability Assessment", use_container_width=True)

    if submitted:
        new_profile = {
            "Gender": gender, "Branch": branch, "CGPA": cgpa, "SSC_Marks": ssc, "HSC_Marks": hsc,
            "Internships": internships, "Projects": projects, "Certifications": certifications,
            "Backlogs": backlogs, "Aptitude_Score": aptitude, "Technical_Skill": technical,
            "Coding_Skill": coding, "Communication_Skill": communication, "Soft_Skill": soft,
            "Extracurricular_Score": extracurricular, "Leadership_Score": leadership,
            "Placement_Training": placement_training, "Workshops_Attended": workshops,
            "Mock_Interview_Score": mock_interview, "LinkedIn_GitHub_Activity": linkedin_github,
        }
        st.session_state["profile"] = new_profile
        st.session_state["student_name"] = name
        st.session_state["whatif_profile"] = new_profile.copy()
        st.session_state["assessment_run"] = True
        st.session_state["chat_history"] = []
        st.session_state["interview_log"] = []
        st.session_state["interview_question"] = None
        st.session_state["pending_page"] = "🏠 Home / Full Report"
        st.rerun()

    if not st.session_state["assessment_run"]:
        st.info("👆 Fill in the form above and click **Run Employability Assessment** to unlock your full "
                "report plus the AI Mentor chatbot, Interview Lab, and Company Readiness tools.")


# ========================================================================================
# ORIGINAL REPORT SECTIONS (logic unchanged — rendered together on Home / Full Report)
# ========================================================================================
def render_overview(profile, student_name):
    st.markdown('<div class="section-title" id="overview">🏠 Overview & Prediction</div>', unsafe_allow_html=True)
    label, probability, package = predict_profile(profile)
    emp_score = compute_employability_score(profile)

    st.markdown(f"### Hi **{student_name}** 👋 — here's your AI-powered employability snapshot")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Placement Probability", f"{probability*100:.1f}%")
        st.markdown(probability_badge(probability), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Employability Score", f"{emp_score:.1f} / 100")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Package", f"₹{package:.2f} LPA" if label else "—")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        percentile = (df_reference["Employability_Score"] < emp_score).mean() * 100
        st.metric("Peer Percentile", f"{percentile:.0f}th")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_gauge, col_dist = st.columns([1, 1.3])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Placement Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(251,113,133,0.25)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.22)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.22)"},
                ],
                "threshold": {"line": {"color": "#eef1fb", "width": 3}, "value": probability * 100},
            },
        ))
        fig.update_layout(height=320, margin=dict(t=50, b=10, l=20, r=20),
                           paper_bgcolor=CHART_PAPER_BG, font_color=CHART_FONT_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        fig2 = px.histogram(df_reference, x="Employability_Score", color="Placed",
                             nbins=40, opacity=0.75,
                             color_discrete_map={0: "#fb7185", 1: "#34d399"},
                             labels={"Placed": "Placed (1) / Not Placed (0)"},
                             title="Where you stand vs. 6,000 peer profiles")
        fig2.add_vline(x=emp_score, line_width=3, line_dash="dash", line_color="#eef1fb",
                        annotation_text="You", annotation_position="top")
        fig2.update_layout(height=320, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                            font_color=CHART_FONT_COLOR, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 Update your profile any time and click **'Run Employability Assessment'** to refresh "
            "every section of this report.")

    with st.expander("📋 Model Card — how this prediction was made"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model", METRICS["best_model"])
        mc2.metric("Accuracy", f"{METRICS['accuracy']*100:.1f}%")
        mc3.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
        mc4.metric("F1 Score", f"{METRICS['f1_score']:.3f}")
        st.caption(f"Trained on {METRICS['n_train']} students, tested on {METRICS['n_test']} held-out students. "
                   f"Package regressor R² = {METRICS.get('package_regressor_r2', 'N/A')} (predicted only for "
                   f"students likely to be placed).")


def render_explainable(profile):
    st.markdown('<div class="section-title" id="explainable">🔍 Why did the model predict this?</div>', unsafe_allow_html=True)
    label, probability, package = predict_profile(profile)
    expl_df, method = get_explanation(profile)
    st.caption(f"Explanation method: **{method}**")

    expl_df_sorted = expl_df.sort_values("Impact")
    colors = ["#34d399" if v > 0 else "#fb7185" for v in expl_df_sorted["Impact"]]
    fig = go.Figure(go.Bar(
        x=expl_df_sorted["Impact"], y=expl_df_sorted["Feature"], orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Top factors influencing YOUR placement probability",
        xaxis_title="Impact on prediction (← decreases | increases →)",
        height=420, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
        font_color=CHART_FONT_COLOR, margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Factors working in your favor")
        positives = expl_df[expl_df["Impact"] > 0].sort_values("Impact", ascending=False)
        if positives.empty:
            st.write("No strongly positive factors yet — see the Improvement Plan section.")
        for _, r in positives.iterrows():
            st.markdown(f"- **{r['Feature'].replace('_',' ')}** is boosting your prediction")
    with col_b:
        st.markdown("#### ⚠️ Factors holding you back")
        negatives = expl_df[expl_df["Impact"] < 0].sort_values("Impact")
        if negatives.empty:
            st.write("No strongly negative factors detected. Great profile!")
        for _, r in negatives.iterrows():
            st.markdown(f"- **{r['Feature'].replace('_',' ')}** is reducing your prediction")

    st.markdown("---")
    st.markdown("#### 🌍 Global feature importance (what matters most across all students)")
    global_imp = pd.DataFrame({
        "Feature": list(METRICS["feature_importances"].keys()),
        "Importance": list(METRICS["feature_importances"].values()),
    }).head(12)
    fig3 = px.bar(global_imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Viridis")
    fig3.update_layout(height=420, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                        font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)


def render_skill_gap(profile):
    st.markdown('<div class="section-title" id="skillgap">🧩 Skill Gap Analysis</div>', unsafe_allow_html=True)
    role_matches = compute_role_match(profile)
    best_role = role_matches[0]

    st.markdown(f"Best-fit role right now: **{best_role['icon']} {best_role['role']}** "
                f"({best_role['match']}% match)")

    radar_categories = ["CGPA", "Technical_Skill", "Coding_Skill", "Communication_Skill",
                         "Aptitude_Score", "Certifications", "Projects"]
    student_vals, target_vals = [], []
    target_spec = JOB_ROLES[best_role["role"]]["weights"]
    for cat in radar_categories:
        raw = profile.get(cat, 0)
        norm = raw / 10 if cat not in ["Aptitude_Score"] else raw / 100
        if cat == "Certifications" or cat == "Projects":
            norm = min(raw / 5, 1.0)
        student_vals.append(norm * 100)
        t = target_spec.get(cat, None)
        if t is not None:
            tnorm = t / 10 if cat not in ["Aptitude_Score"] else t / 100
            if cat in ["Certifications", "Projects"]:
                tnorm = min(t / 5, 1.0)
            target_vals.append(tnorm * 100)
        else:
            target_vals.append(60)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=student_vals, theta=radar_categories, fill="toself",
                                   name="You", line_color="#8b5cf6"))
    fig.add_trace(go.Scatterpolar(r=target_vals, theta=radar_categories, fill="toself",
                                   name=f"{best_role['role']} target", line_color="#fbbf24",
                                   opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                       height=460, paper_bgcolor=CHART_PAPER_BG, font_color=CHART_FONT_COLOR,
                       legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📉 Specific gaps to close")
    if best_role["gaps"]:
        gap_df = pd.DataFrame(best_role["gaps"])
        gap_df.columns = ["Skill", "Your Level", "Required Level", "Gap"]
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

        fig_gap = px.bar(gap_df, x="Gap", y="Skill", orientation="h", color="Gap",
                          color_continuous_scale="Reds")
        fig_gap.update_layout(height=350, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                               font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.success("🎉 You already meet or exceed every requirement for this role!")

    st.markdown("#### 🧠 Core skills expected for this role")
    st.markdown("".join([f'<span class="pill">{s}</span>' for s in JOB_ROLES[best_role["role"]]["core_skills"]]),
                unsafe_allow_html=True)


def render_job_matching(profile):
    st.markdown('<div class="section-title" id="jobmatch">🎯 Job Role Matching</div>', unsafe_allow_html=True)
    st.caption("How well your current profile fits common entry-level job roles.")

    role_matches = compute_role_match(profile)
    match_df = pd.DataFrame([{"Role": r["icon"] + " " + r["role"], "Match %": r["match"]} for r in role_matches])
    fig = px.bar(match_df, x="Match %", y="Role", orientation="h", range_x=[0, 100],
                 color="Match %", color_continuous_scale="Tealgrn", text="Match %")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=380, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                       font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=60))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explore each role")
    role_cols = st.columns(3)
    for i, r in enumerate(role_matches):
        with role_cols[i % 3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### {r['icon']} {r['role']}")
            st.progress(min(int(r["match"]), 100))
            st.markdown(f"**{r['match']}% match**")
            st.caption(r["description"])
            st.markdown("**Core skills:** " + ", ".join(r["core_skills"][:3]) + "...")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def render_resume_analysis(profile):
    st.markdown('<div class="section-title" id="resume">📄 Resume Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload your resume (PDF or TXT) to extract skills and check coverage automatically.")

    uploaded = st.file_uploader("Upload Resume", type=["pdf", "txt"])
    if uploaded is not None:
        with st.spinner("Reading and analyzing your resume..."):
            text = extract_resume_text(uploaded)
            found, coverage, word_count = analyze_resume(text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Skill Keyword Coverage", f"{coverage}%")
        c2.metric("Word Count", word_count)
        total_found = sum(len(v) for v in found.values())
        c3.metric("Skills/Keywords Detected", total_found)

        st.markdown("#### 🗂️ Detected skills by category")
        for cat, kws in found.items():
            if kws:
                st.markdown(f"**{cat}:** " + "".join([f'<span class="pill">{k}</span>' for k in kws]),
                             unsafe_allow_html=True)
            else:
                st.markdown(f"**{cat}:** _none detected_")

        cat_df = pd.DataFrame({
            "Category": list(found.keys()),
            "Keywords Found": [len(v) for v in found.values()],
        })
        fig = px.bar(cat_df, x="Category", y="Keywords Found", color="Keywords Found",
                     color_continuous_scale="Purp")
        fig.update_layout(height=350, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                           font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        if word_count < 150:
            st.warning("⚠️ Your resume looks short. Aim for a focused single page (300-500 words) "
                       "covering education, projects, skills, and experience.")
        if coverage < 25:
            st.warning("⚠️ Low keyword coverage — consider explicitly listing technical tools/skills "
                       "you've used (many resumes are first filtered by keyword scanners).")
        else:
            st.success("✅ Your resume shows solid keyword coverage for automated screening (ATS) systems.")

        if st.button("↪️ Use these resume skills to boost my Technical Skill estimate"):
            boost = min(10.0, profile["Technical_Skill"] + total_found * 0.15)
            st.session_state["profile"]["Technical_Skill"] = round(boost, 1)
            st.success(f"Technical Skill updated to {boost:.1f}/10 based on resume content. "
                       "Refresh (re-run) to see the updated prediction everywhere.")
    else:
        st.info("👆 Upload a resume to get an instant, automated skills breakdown.")


def render_improvement_plan(profile, student_name):
    st.markdown('<div class="section-title" id="improve">📌 Personalized Improvement Plan</div>', unsafe_allow_html=True)
    label, probability, package = predict_profile(profile)
    role_matches = compute_role_match(profile)
    plan = generate_improvement_plan(profile, probability, role_matches[0]["gaps"])

    st.caption(f"Generated specifically for **{student_name}** based on current profile "
               f"(placement probability: {probability*100:.1f}%).")

    priority_colors = {"Critical": "badge-red", "High": "badge-red", "Medium": "badge-yellow", "Low": "badge-green"}
    for category, title, detail, priority in plan:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            badge_class = priority_colors.get(priority, "badge-yellow")
            st.markdown(f'<span class="badge {badge_class}">{priority} priority</span> '
                        f'<span class="pill">{category}</span>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(detail)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📅 Suggested 30-day action sprint")
    sprint = [
        ("Week 1", "Audit & fix fundamentals", "Clear backlogs if any, revise weak subjects, set up LeetCode/HackerRank profile."),
        ("Week 2", "Build/polish 1 project", "Ship one solid, demo-ready project and push it to GitHub with a clean README."),
        ("Week 3", "Certifications + aptitude", "Complete 1 certification and do daily aptitude practice (30 mins)."),
        ("Week 4", "Interview readiness", "2 mock interviews in the Interview Lab, polish resume, update LinkedIn, apply to 10+ roles."),
    ]
    sprint_cols = st.columns(4)
    for i, (week, title, desc) in enumerate(sprint):
        with sprint_cols[i]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{week}**")
            st.markdown(f"*{title}*")
            st.caption(desc)
            st.markdown('</div>', unsafe_allow_html=True)


def render_whatif_simulator(profile):
    st.markdown('<div class="section-title" id="whatif">🎛️ What-If Simulator</div>', unsafe_allow_html=True)
    st.caption("Drag the sliders to see how improving specific factors changes your placement probability "
               "— in real time, without touching your saved profile.")

    base_label, base_prob, base_package = predict_profile(profile)

    if st.button("🔄 Reset simulator to my current profile"):
        st.session_state["whatif_profile"] = profile.copy()

    wp = st.session_state["whatif_profile"]

    sim_cols = st.columns(2)
    with sim_cols[0]:
        wp["CGPA"] = st.slider("Simulated CGPA", 4.0, 10.0, float(wp["CGPA"]), 0.05, key="sim_cgpa")
        wp["Internships"] = st.slider("Simulated Internships", 0, 6, int(wp["Internships"]), key="sim_intern")
        wp["Projects"] = st.slider("Simulated Projects", 0, 10, int(wp["Projects"]), key="sim_proj")
        wp["Certifications"] = st.slider("Simulated Certifications", 0, 10, int(wp["Certifications"]), key="sim_cert")
        wp["Backlogs"] = st.slider("Simulated Backlogs", 0, 6, int(wp["Backlogs"]), key="sim_back")
    with sim_cols[1]:
        wp["Aptitude_Score"] = st.slider("Simulated Aptitude Score", 0, 100, int(wp["Aptitude_Score"]), key="sim_apt")
        wp["Technical_Skill"] = st.slider("Simulated Technical Skill", 0.0, 10.0, float(wp["Technical_Skill"]), 0.1, key="sim_tech")
        wp["Coding_Skill"] = st.slider("Simulated Coding Skill", 0.0, 10.0, float(wp["Coding_Skill"]), 0.1, key="sim_code")
        wp["Communication_Skill"] = st.slider("Simulated Communication Skill", 0.0, 10.0, float(wp["Communication_Skill"]), 0.1, key="sim_comm")
        wp["Placement_Training"] = st.selectbox("Simulated Placement Training", ["Yes", "No"],
                                                 index=0 if wp["Placement_Training"] == "Yes" else 1, key="sim_train")

    st.session_state["whatif_profile"] = wp
    sim_label, sim_prob, sim_package = predict_profile(wp)

    delta = (sim_prob - base_prob) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Probability", f"{base_prob*100:.1f}%")
    c2.metric("Simulated Probability", f"{sim_prob*100:.1f}%", delta=f"{delta:+.1f} pts")
    c3.metric("Simulated Package", f"₹{sim_package:.2f} LPA" if sim_label else "—")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Current", "Simulated"], y=[base_prob*100, sim_prob*100],
                          marker_color=["#8b5cf6", "#34d399" if delta >= 0 else "#fb7185"],
                          text=[f"{base_prob*100:.1f}%", f"{sim_prob*100:.1f}%"], textposition="outside"))
    fig.update_layout(height=350, yaxis_range=[0, 100], paper_bgcolor=CHART_PAPER_BG,
                       plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    if delta > 0:
        st.success(f"📈 These changes could raise your placement probability by **{delta:.1f} percentage points**.")
    elif delta < 0:
        st.warning(f"📉 These changes would lower your placement probability by **{abs(delta):.1f} percentage points**.")
    else:
        st.info("No change in probability yet — try adjusting a few sliders.")

    if st.button("💾 Apply simulated values to my main profile"):
        st.session_state["profile"] = wp.copy()
        st.success("Applied! Scroll up to the Overview & Prediction section to see your updated assessment.")


def render_progress_tracking(profile, student_name):
    st.markdown('<div class="section-title" id="progress">📈 Progress Tracking</div>', unsafe_allow_html=True)
    st.caption("Save periodic snapshots of your assessment to track improvement over time.")

    label, probability, package = predict_profile(profile)
    snapshot_emp_score = compute_employability_score(profile)
    if st.button("📌 Save today's snapshot"):
        log = save_progress_entry(student_name, probability, snapshot_emp_score, package)
        st.success("Snapshot saved!")

    log = load_progress_log()
    my_log = log[log["student_name"] == student_name] if not log.empty else log

    if my_log.empty:
        st.info("No snapshots yet for this name. Click **'Save today's snapshot'** above to start tracking, "
                "then come back after future assessments to see your trend.")
    else:
        my_log["timestamp"] = pd.to_datetime(my_log["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=my_log["timestamp"], y=my_log["probability"], mode="lines+markers",
                                  name="Placement Probability (%)", line=dict(color="#8b5cf6", width=3)))
        fig.add_trace(go.Scatter(x=my_log["timestamp"], y=my_log["employability_score"], mode="lines+markers",
                                  name="Employability Score", line=dict(color="#34d399", width=3)))
        fig.update_layout(height=420, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                           font_color=CHART_FONT_COLOR, legend=dict(orientation="h", y=-0.15),
                           margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📜 Snapshot history")
        st.dataframe(my_log[["timestamp", "probability", "employability_score", "package"]]
                     .sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        if len(my_log) >= 2:
            first, last = my_log.iloc[0], my_log.iloc[-1]
            improvement = last["probability"] - first["probability"]
            if improvement > 0:
                st.success(f"🎉 You've improved your placement probability by **{improvement:.1f} points** "
                           f"since your first snapshot!")
            elif improvement < 0:
                st.warning(f"Your placement probability dropped by {abs(improvement):.1f} points since your "
                           f"first snapshot — check the Improvement Plan section.")


def render_full_report(profile, student_name):
    """Renders every original feature, one after another, automatically."""
    st.markdown(
        "".join([
            '<a class="toc-pill" href="#overview">🏠 Overview</a>',
            '<a class="toc-pill" href="#explainable">🔍 Explainable AI</a>',
            '<a class="toc-pill" href="#skillgap">🧩 Skill Gap</a>',
            '<a class="toc-pill" href="#jobmatch">🎯 Job Matching</a>',
            '<a class="toc-pill" href="#resume">📄 Resume</a>',
            '<a class="toc-pill" href="#improve">📌 Improvement Plan</a>',
            '<a class="toc-pill" href="#whatif">🎛️ What-If Simulator</a>',
            '<a class="toc-pill" href="#progress">📈 Progress</a>',
        ]),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    render_overview(profile, student_name)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_explainable(profile)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_skill_gap(profile)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_job_matching(profile)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_resume_analysis(profile)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_improvement_plan(profile, student_name)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_whatif_simulator(profile)
    st.markdown('<hr class="report-divider">', unsafe_allow_html=True)

    render_progress_tracking(profile, student_name)


# ========================================================================================
# 🤖 ASK YOUR MENTOR — page
# ========================================================================================
def render_mentor_chat(profile, student_name):
    st.markdown('<div class="section-title">🤖 Ask Your Mentor</div>', unsafe_allow_html=True)
    st.caption("A personalized AI mentor that answers questions using YOUR profile data — instantly, offline, no waiting. "
               "It now understands a much wider range of questions, comparisons, and general placement terms.")

    if not st.session_state["chat_history"]:
        st.session_state["chat_history"].append(("assistant", _mentor_greeting(student_name)))

    st.markdown("**💬 Quick questions**")
    qcols = st.columns(3)
    quick_qs = [
        "Am I ready for placements?",
        "What should I improve first?",
        "What are my strengths?",
        "Which companies should I target?",
        "Which job role suits me best?",
        "What package can I expect?",
        "How's my resume looking?",
        "What's my roadmap for the next month?",
        "What is DSA?",
    ]
    for i, qq in enumerate(quick_qs):
        with qcols[i % 3]:
            if st.button(qq, key=f"quick_q_{i}", use_container_width=True):
                st.session_state["pending_chat_msg"] = qq

    if st.session_state.get("pending_chat_msg"):
        msg = st.session_state["pending_chat_msg"]
        st.session_state["pending_chat_msg"] = None
        st.session_state["chat_history"].append(("user", msg))
        reply = generate_mentor_reply(msg, profile, student_name)
        st.session_state["chat_history"].append(("assistant", reply))

    st.markdown("---")
    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    user_msg = st.chat_input("Type your question here (e.g. 'What should I improve?')")
    if user_msg:
        st.session_state["chat_history"].append(("user", user_msg))
        reply = generate_mentor_reply(user_msg, profile, student_name)
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()


# ========================================================================================
# 🏢 COMPANY READINESS — page
# ========================================================================================
def render_company_readiness(profile):
    st.markdown('<div class="section-title">🏢 Company-Specific Readiness Score</div>', unsafe_allow_html=True)
    st.caption("How ready is your current profile for real hiring bars at 9 well-known companies?")

    results = compute_company_readiness(profile)

    best = results[0]
    st.markdown(f"🏆 **Best current fit:** {best['icon']} **{best['name']}** — "
                f"**{best['readiness_pct']}%** ready ({best['status_label']})")

    comp_df = pd.DataFrame([{"Company": f"{r['icon']} {r['name']}", "Readiness %": r["readiness_pct"]} for r in results])
    fig = px.bar(comp_df, x="Readiness %", y="Company", orientation="h", range_x=[0, 100],
                 color="Readiness %", color_continuous_scale="Tealgrn", text="Readiness %")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=420, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                       font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=60))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Full breakdown")
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            st.markdown(f"### {r['icon']} {r['name']}")
            st.caption(r["tier"])
            st.progress(min(int(r["readiness_pct"]), 100))
            st.markdown(f'<span class="badge {r["badge_class"]}">{r["status_label"]} — {r["readiness_pct"]}%</span>',
                        unsafe_allow_html=True)
            if r["gaps"]:
                st.markdown("**To close the gap:**")
                for g in r["gaps"][:3]:
                    st.markdown(f"- {g}")
            else:
                st.markdown("_You meet every criterion!_ 🎉")
            st.markdown('</div>', unsafe_allow_html=True)

    st.info("💡 Want to close these gaps hands-on? Head to the **🎤 Interview Lab** to practice the exact "
            "kind of questions these companies ask, or ask your **🤖 Mentor** for a step-by-step plan.")


# ========================================================================================
# 🎤 INTERVIEW LAB — page
# ========================================================================================
def render_interview_lab(profile):
    st.markdown('<div class="section-title">🎤 Interview Lab</div>', unsafe_allow_html=True)
    st.caption("Practice real mock-interview questions and get instant, structured feedback — no scheduling, "
               "no waiting. Pick a category, answer in your own words, and see exactly what to improve.")

    role_matches = compute_role_match(profile)
    best_role = role_matches[0]

    category_options = {
        "HR / Behavioral": HR_BEHAVIORAL_QUESTIONS,
        "Technical Fundamentals": TECHNICAL_FUNDAMENTAL_QUESTIONS,
        "Aptitude & Reasoning": APTITUDE_REASONING_QUESTIONS,
        f"Role-Specific ({best_role['role']})": _role_specific_questions(best_role),
    }

    cat_cols = st.columns(4)
    labels = list(category_options.keys())
    for i, label in enumerate(labels):
        with cat_cols[i]:
            if st.button(label, key=f"cat_btn_{i}", use_container_width=True):
                st.session_state["interview_category"] = label
                st.session_state["interview_question"] = random.choice(category_options[label])

    if st.session_state["interview_category"] is None:
        st.info("👆 Pick a category above to get your first mock-interview question.")
        return

    active_label = st.session_state["interview_category"]
    active_bank = category_options.get(active_label, HR_BEHAVIORAL_QUESTIONS)
    # keep category eval bucket simple: treat any non-"HR / Behavioral" as technical-style scoring
    eval_category = "HR / Behavioral" if active_label == "HR / Behavioral" else "Technical"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="interview-card">', unsafe_allow_html=True)
    st.markdown(f'<span class="pill">{active_label}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="interview-question">🎙️ {st.session_state["interview_question"]}</div>', unsafe_allow_html=True)

    answer = st.text_area("Your answer", height=150, key=f"answer_box_{st.session_state['interview_question']}",
                           placeholder="Type your answer as if you were speaking it out loud in an interview...")

    b1, b2 = st.columns([1, 1])
    with b1:
        submit = st.button("✅ Submit & Get Feedback", use_container_width=True)
    with b2:
        next_q = st.button("➡️ Skip to Next Question", use_container_width=True)

    if next_q:
        st.session_state["interview_question"] = random.choice(active_bank)
        st.rerun()

    if submit:
        score, tips = evaluate_answer(eval_category, answer)
        st.session_state["interview_log"].append({
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": active_label,
            "question": st.session_state["interview_question"],
            "score": score,
        })

        badge_class = "badge-green" if score >= 8 else ("badge-yellow" if score >= 5.5 else "badge-red")
        st.markdown(f'<span class="badge {badge_class}">Score: {score}/10</span>', unsafe_allow_html=True)
        st.markdown("#### 📝 Feedback")
        for tip in tips:
            st.markdown(f"- {tip}")

        if st.button("➡️ Try another question in this category"):
            st.session_state["interview_question"] = random.choice(active_bank)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- session trend ----
    if st.session_state["interview_log"]:
        st.markdown("---")
        st.markdown("#### 📈 Your Interview Lab session")
        log_df = pd.DataFrame(st.session_state["interview_log"])
        avg_score = log_df["score"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Questions Attempted", len(log_df))
        m2.metric("Average Score", f"{avg_score:.1f}/10")
        m3.metric("Best Category", log_df.groupby("category")["score"].mean().idxmax())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(log_df) + 1)), y=log_df["score"], mode="lines+markers",
                                  name="Score", line=dict(color="#22d3ee", width=3)))
        fig.update_layout(height=320, yaxis_range=[0, 10], xaxis_title="Attempt #", yaxis_title="Score",
                           paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
                           font_color=CHART_FONT_COLOR, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📜 Full attempt history"):
            st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        if st.button("🔄 Reset Interview Lab session"):
            st.session_state["interview_log"] = []
            st.rerun()


# --------------------------------------------------------------------------------------
# ROUTING
# --------------------------------------------------------------------------------------
if nav == "📝 Student Profile":
    render_profile_page()
else:
    if not st.session_state["assessment_run"]:
        render_header("Complete your **Student Profile** first to unlock this section.")
        st.warning("🔒 This section unlocks after you fill in your Student Profile and click "
                   "**'Run Employability Assessment'**.")
        if st.button("📝 Go to Student Profile"):
            st.session_state["pending_page"] = "📝 Student Profile"
            st.rerun()
        st.stop()

    profile = st.session_state["profile"]
    student_name = st.session_state.get("student_name", "Student")

    if nav == "🏠 Home / Full Report":
        render_header(f"Welcome back, **{student_name}** — your complete assessment is below, generated automatically.")
        render_full_report(profile, student_name)
    elif nav == "🤖 Ask Your Mentor":
        render_mentor_chat(profile, student_name)
    elif nav == "🎤 Interview Lab":
        render_interview_lab(profile)
    elif nav == "🏢 Company Readiness":
        render_company_readiness(profile)
