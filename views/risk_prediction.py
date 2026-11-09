import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import app_helpers as H

H.require_artifacts()
import utils.explainability as xai

art = H.load_art()
meta = art["meta"]
labels = meta["category_labels"]
ranges = meta["numeric_ranges"]
threshold = float(meta.get("default_threshold", 0.5))
columns = meta["feature_columns"]
data = H.load_data()

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# ------------------------------------------------------------------ presets (stored as raw codes)
RISKY = {"age": 62.0, "sex": 1, "cp": 3, "trestbps": 150.0, "chol": 270.0, "fbs": 0, "restecg": 2,
         "thalach": 110.0, "exang": 1, "oldpeak": 3.0, "slope": 1, "ca": 2, "thal": 2}
LOWER = {"age": 40.0, "sex": 0, "cp": 2, "trestbps": 118.0, "chol": 190.0, "fbs": 0, "restecg": 0,
         "thalach": 178.0, "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0}


def average_patient():
    p = {}
    for c in NUMERIC:
        p[c] = float(data[c].median())
    for c in CATEGORICAL:
        p[c] = int(data[c].mode().iloc[0])
    return p


PRESETS = {
    "Average patient (dataset medians)": average_patient(),
    "Example: higher-risk profile": RISKY,
    "Example: lower-risk profile": LOWER,
}


def apply_preset(name):
    values = PRESETS[name]
    for col, v in values.items():
        st.session_state[f"in_{col}"] = v
    st.session_state["assessed"] = dict(values)


if "in_age" not in st.session_state:
    apply_preset("Average patient (dataset medians)")


def collect_inputs():
    p = {}
    for c in NUMERIC:
        p[c] = float(st.session_state[f"in_{c}"])
    for c in CATEGORICAL:
        p[c] = int(st.session_state[f"in_{c}"])
    return p


def code_select(label, col, help_text):
    options = [int(k) for k in labels[col].keys()]
    return st.selectbox(label, options, key=f"in_{col}", help=help_text,
                        format_func=lambda code, col=col: labels[col][str(code)])


# ------------------------------------------------------------------ page
st.title("❤️ Risk Prediction")
st.markdown("Fill in the patient's details, then press **Assess risk**. Or start from an example profile.")
H.disclaimer()

st.selectbox("Start from", list(PRESETS.keys()), key="preset",
             on_change=lambda: apply_preset(st.session_state["preset"]))

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Patient and vitals**")
        st.number_input("Age (years)", 18.0, 100.0, step=1.0, format="%.0f", key="in_age")
        code_select("Sex", "sex", "Biological sex.")
        st.number_input("Resting blood pressure (mm Hg)", 70.0, 250.0, step=1.0, format="%.0f", key="in_trestbps",
                        help="Blood pressure at rest (the 'top number').")
        st.number_input("Cholesterol (mg/dl)", 100.0, 600.0, step=1.0, format="%.0f", key="in_chol",
                        help="Serum cholesterol.")
        code_select("Fasting blood sugar", "fbs", "Whether fasting blood sugar is above 120 mg/dl.")
    with c2:
        st.markdown("**Symptoms and exercise test**")
        code_select("Chest pain type", "cp", "Type of chest pain the patient reports.")
        st.number_input("Max heart rate reached (bpm)", 50.0, 230.0, step=1.0, format="%.0f", key="in_thalach",
                        help="Highest heart rate reached during the exercise stress test.")
        code_select("Exercise-induced angina", "exang", "Did exercise trigger chest pain?")
        st.number_input("ST depression", 0.0, 8.0, step=0.1, format="%.1f", key="in_oldpeak",
                        help="How far the ECG's ST segment dropped during exercise compared with rest.")
        code_select("ST slope at peak exercise", "slope", "Slope of the ST segment at peak exercise.")
    with c3:
        st.markdown("**Tests**")
        code_select("Resting ECG", "restecg", "Resting electrocardiogram result.")
        code_select("Major vessels colored", "ca", "Number of major blood vessels (0-3) colored by fluoroscopy.")
        code_select("Thal test", "thal", "Result of the thallium stress test.")
    submitted = st.form_submit_button("Assess risk", type="primary")

if submitted or "assessed" not in st.session_state:
    st.session_state["assessed"] = collect_inputs()
patient = st.session_state["assessed"]

# warn when inputs fall outside what the model saw during training
outside = [f"{xai.PRETTY[c]} = {patient[c]:g}" for c in NUMERIC
           if patient[c] < ranges[c]["min"] or patient[c] > ranges[c]["max"]]
if outside:
    st.warning("Outside the range seen in the training data: " + ", ".join(outside) +
               ". The model has little experience with values like these, so treat the score with extra caution.")

result = xai.explain_patient(art, patient)
risk = result["risk"]
avg_score = float(art["model"].predict_proba(art["X_train"])[:, 1].mean())
flagged = risk >= threshold

st.divider()
left, right = st.columns([1, 1.15])

with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%", "font": {"size": 54}},
        title={"text": "Model-estimated risk score"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": H.RED if flagged else H.BLUE, "thickness": 0.35},
            "steps": [{"range": [0, threshold * 100], "color": "rgba(46,134,171,0.15)"},
                      {"range": [threshold * 100, 100], "color": "rgba(192,57,43,0.15)"}],
            "threshold": {"line": {"color": "#333333", "width": 3}, "thickness": 0.85, "value": threshold * 100},
        },
    ))
    gauge.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=10))
    H.show_plot(gauge, key="gauge")

with right:
    badge = ('<span class="hr-badge high">Higher estimated risk</span>' if flagged
             else '<span class="hr-badge low">Lower estimated risk</span>')
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown(
        f"The model flags **higher estimated risk** when the score is at or above {threshold:.0%} "
        f"(the line on the gauge)."
    )
    m1, m2 = st.columns(2)
    m1.metric("Score", f"{risk:.0%}")
    m2.metric("Average in training data", f"{avg_score:.0%}", delta=f"{(risk - avg_score) * 100:+.0f} pts vs average",
              delta_color="off")
    st.markdown(f"**In words:** {xai.explain_sentence(result)}")
    st.caption("The score is the model's estimate, not a calibrated clinical probability, and not a diagnosis.")

# ------------------------------------------------------------------ explanation
st.subheader("Why this score?")
unit_prob = art["scale"] == "probability"
top = result["table"].head(8).iloc[::-1]
vals = top["shap"] * (100 if unit_prob else 1)
fig = go.Figure(go.Bar(
    x=vals, y=top["feature"], orientation="h",
    marker_color=[H.RED if v > 0 else H.BLUE for v in vals],
    text=[f"{v:+.1f}" if unit_prob else f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
))
fig.add_vline(x=0, line_width=1, line_color=H.GREY)
fig.update_layout(
    height=420, margin=dict(l=10, r=40, t=20, b=40),
    xaxis_title=("Effect on the score (percentage points)" if unit_prob else "Effect on the score (log-odds)"),
    yaxis_title=None,
)
H.show_plot(fig, key="contrib")
st.caption("Red bars pushed the score up, blue bars pushed it down, starting from the average patient. "
           "These are SHAP values: they describe how the model behaves, not medical cause and effect.")

tab_factors, tab_whatif, tab_waterfall, tab_download = st.tabs(
    ["All factors", "What-if explorer", "SHAP waterfall", "Download summary"])

with tab_factors:
    table = result["table"].copy()
    table["effect"] = table["shap"].apply(lambda v: "pushes score up" if v > 0 else "pushes score down")
    table["shap"] = table["shap"] * (100 if unit_prob else 1)
    table = table.rename(columns={"feature": "Feature = value",
                                  "shap": "Effect (points)" if unit_prob else "Effect (log-odds)",
                                  "effect": "Direction"})
    st.dataframe(table.round(2), hide_index=True)

with tab_whatif:
    st.markdown("Change one measurement and keep everything else fixed to see how the score responds.")
    feature = st.selectbox("Measurement to vary", NUMERIC, index=NUMERIC.index("thalach"),
                           format_func=lambda c: xai.PRETTY[c])
    grid = np.linspace(ranges[feature]["min"], ranges[feature]["max"], 40)
    rows = pd.DataFrame([patient] * len(grid))
    rows[feature] = grid
    curve = art["model"].predict_proba(rows[columns])[:, 1] * 100
    wf = go.Figure()
    wf.add_trace(go.Scatter(x=grid, y=curve, mode="lines", line=dict(color=H.RED, width=3), name="Score"))
    wf.add_trace(go.Scatter(x=[patient[feature]], y=[risk * 100], mode="markers", name="This patient",
                            marker=dict(size=13, color="#222222", line=dict(width=2, color="white"))))
    wf.add_hline(y=threshold * 100, line_dash="dot", line_color=H.GREY,
                 annotation_text="flag threshold", annotation_position="bottom right")
    wf.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40), xaxis_title=xai.PRETTY[feature],
                     yaxis_title="Model-estimated risk score (%)", yaxis_range=[0, 100],
                     legend=dict(orientation="h", y=1.1))
    H.show_plot(wf, key="whatif")
    st.caption(f"The curve covers the range of {xai.PRETTY[feature].lower()} seen in the training data "
               f"({ranges[feature]['min']:g} to {ranges[feature]['max']:g}).")

with tab_waterfall:
    fig_wf = xai.plot_patient(art, result, max_display=10)
    st.pyplot(fig_wf)
    st.caption("Standard SHAP waterfall: starts at the average score and adds each feature's effect.")

with tab_download:
    lines = ["HeartRisk AI - patient summary", "(Educational tool, not a medical diagnosis)", "",
             "Inputs:"]
    for c in columns:
        shown = labels[c][str(int(patient[c]))] if c in labels else f"{patient[c]:g}"
        lines.append(f"  - {xai.PRETTY.get(c, c)}: {shown}")
    lines += ["", f"Model-estimated risk score: {risk:.0%}",
              f"Flag threshold: {threshold:.0%} -> {'higher' if flagged else 'lower'} estimated risk", "",
              xai.explain_sentence(result), "",
              f"Model: {meta['model']}",
              "The score is not a calibrated clinical probability."]
    report = "\n".join(lines)
    st.text(report)
    st.download_button("Download as text file", report, file_name="heartrisk_summary.txt", mime="text/plain")
