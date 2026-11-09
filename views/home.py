import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from utils import app_helpers as H

H.require_artifacts()
meta = H.load_meta()
data = H.load_data()
tm = meta["test_metrics"]


def link(path, label, icon):
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:          # e.g. when a page is run on its own
        st.caption(f"{icon} {label}")


st.markdown(
    '<div class="hr-hero"><h1>❤️ HeartRisk AI</h1>'
    '<div class="hr-sub">Heart disease risk estimation with explainable machine learning</div></div>',
    unsafe_allow_html=True,
)
st.write("")
st.markdown(
    "Enter a patient's clinical details and the app returns a **model-estimated risk score** together with "
    "a plain-language explanation of **why** the model gave that score, using SHAP. It also lets you compare "
    "the models, inspect how they were evaluated, and explore the data behind them."
)
link("views/risk_prediction.py", "Try the risk predictor", "➡️")

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in dataset", f"{len(data)}")
c2.metric("Clinical features", f"{len(meta['feature_columns'])}")
c3.metric("Test ROC-AUC", f"{tm['roc_auc']:.2f}")
c4.metric("Test recall (disease)", f"{tm['recall_sensitivity']:.2f}")
st.caption(
    "Test metrics come from a single evaluation on 60 held-out patients that were never used for training "
    "or model selection. With a sample this small they are rough estimates."
)

st.write("")
st.subheader("How it works")
a, b, c = st.columns(3)
a.markdown(
    '<div class="hr-card"><h4>1 · Enter details</h4>'
    "Age, blood pressure, cholesterol, exercise-test results and other clinical measurements, "
    "using plain-language options instead of codes.</div>",
    unsafe_allow_html=True,
)
b.markdown(
    '<div class="hr-card"><h4>2 · Get a risk score</h4>'
    "A machine-learning pipeline preprocesses the inputs and returns a score between 0% and 100%. "
    "It is an estimate, not a calibrated medical probability.</div>",
    unsafe_allow_html=True,
)
c.markdown(
    '<div class="hr-card"><h4>3 · See why</h4>'
    "SHAP breaks the score into the contribution of each feature, so you can see what pushed it up "
    "or down for this patient.</div>",
    unsafe_allow_html=True,
)

st.write("")
st.subheader("What is inside")
left, right = st.columns(2)
with left:
    st.markdown(
        "- **Risk Prediction:** score, explanation, what-if explorer and a downloadable summary\n"
        "- **Model Comparison:** metrics, ROC and precision-recall curves, threshold trade-offs\n"
        "- **Data Insights:** interactive exploration of the dataset with filters"
    )
with right:
    st.markdown(
        "- **Honest evaluation:** stratified train/test split, cross-validated tuning, test set used once\n"
        "- **Two preprocessing pipelines:** one for linear models, one for tree models\n"
        "- **Clinically motivated feature:** heart rate relative to the age-expected maximum"
    )

st.write("")
st.subheader("Built with")
st.markdown("`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Plotly` · `Streamlit`")

st.write("")
H.disclaimer()
