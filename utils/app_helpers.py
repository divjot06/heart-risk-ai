"""Shared helpers for the HeartRisk AI Streamlit app (paths, cached loaders, styling)."""
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # lets the saved pipeline import utils.preprocessing
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "models" / "heart_cleveland"
DATA_PATH = ROOT / "data" / "heart_cleveland_clean.csv"

RED = "#C0392B"
BLUE = "#2E86AB"
GREY = "#8A8F98"
TARGET_NAMES = {0: "No disease", 1: "Disease"}
TARGET_COLORS = {"No disease": BLUE, "Disease": RED}

REQUIRED_FILES = [
    ARTIFACT_DIR / "heart_risk_pipeline.joblib",
    ARTIFACT_DIR / "metadata.json",
    ARTIFACT_DIR / "X_train.csv",
    ARTIFACT_DIR / "model_comparison.csv",
    ARTIFACT_DIR / "threshold_table.csv",
    DATA_PATH,
    ROOT / "utils" / "preprocessing.py",
    ROOT / "utils" / "explainability.py",
]


# ----------------------------------------------------------------------------- styling
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1180px;}
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.25);
            padding: 14px 16px; border-radius: 12px;
        }
        .hr-badge {display:inline-block; padding:6px 16px; border-radius:999px;
                   font-weight:600; font-size:0.95rem; margin-bottom:6px;}
        .hr-badge.high {background:#FDEDEC; color:#A93226; border:1px solid #F5B7B1;}
        .hr-badge.low  {background:#EAF2F8; color:#1F618D; border:1px solid #AED6F1;}
        .hr-card {border:1px solid rgba(128,128,128,0.25); border-radius:14px;
                  padding:18px 20px; background: rgba(128,128,128,0.05); height:100%;}
        .hr-card h4 {margin-top:0;}
        .hr-hero {padding: 6px 0 2px 0;}
        .hr-hero h1 {font-size: 2.6rem; margin-bottom: 0.2rem;}
        .hr-sub {font-size: 1.15rem; opacity: 0.85;}
        .hr-note {font-size: 0.85rem; opacity: 0.75;}
        </style>
        """,
        unsafe_allow_html=True,
    )


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_plot(fig, key=None):
    """st.plotly_chart that works on old and new Streamlit (width='stretch' vs use_container_width)."""
    fig.update_yaxes(automargin=True)          # keep long labels from being cut off
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def disclaimer():
    st.warning(
        "**Educational tool, not a medical device.** The score is an estimate from a machine-learning model "
        "trained on a small public dataset (297 patients). It is not a diagnosis and must not be used to make "
        "medical decisions. If you are worried about your health, please talk to a doctor.",
        icon="⚠️",
    )


# ----------------------------------------------------------------------------- loading
def missing_files():
    return [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]


def require_artifacts():
    """Show a friendly message (instead of a traceback) if the notebook outputs are not in place."""
    missing = missing_files()
    if missing:
        st.error("The app could not find files that the notebook creates.")
        st.markdown(
            "Run `heart_disease_analysis_final.ipynb` from top to bottom (**Run All**) so it writes "
            "the model files, then copy the `models/`, `utils/` and `data/` folders next to `app.py`.\n\n"
            "Missing:\n" + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_art():
    import utils.explainability as xai
    return xai.load_artifacts(str(ARTIFACT_DIR))


@st.cache_data(show_spinner=False)
def load_meta():
    with open(ARTIFACT_DIR / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_comparison():
    return pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_thresholds():
    return pd.read_csv(ARTIFACT_DIR / "threshold_table.csv")


@st.cache_data(show_spinner=False)
def load_test_predictions():
    path = ARTIFACT_DIR / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Computing feature importance...")
def global_importance_table():
    import utils.explainability as xai
    return xai.global_importance(load_art())


def sidebar_info():
    meta = load_meta()
    with st.sidebar:
        st.markdown("### ❤️ HeartRisk AI")
        st.caption("Heart disease risk estimation with explainable machine learning.")
        st.divider()
        st.caption(f"**Model:** {meta['model']}")
        st.caption("**Data:** UCI Cleveland heart disease dataset (297 patients)")
        st.caption("**Explainability:** SHAP")
        st.divider()
        st.caption("Educational tool. Not a medical diagnosis.")


PRETTY_FEATURES = {
    "age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "thalach": "Max heart rate", "oldpeak": "ST depression", "sex": "Sex",
    "fbs": "Fasting blood sugar > 120", "exang": "Exercise-induced angina",
    "cp": "Chest pain type", "restecg": "Resting ECG", "slope": "ST slope",
    "ca": "Major vessels", "thal": "Thal test", "target": "Heart disease",
}
