import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc as area_under_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import app_helpers as H

H.require_artifacts()
meta = H.load_meta()
comparison = H.load_comparison()
thresholds = H.load_thresholds()
test_pred = H.load_test_predictions()
tm = meta["test_metrics"]
default_thr = float(meta.get("default_threshold", 0.5))

st.title("📊 Model Comparison")
st.markdown(
    "Three model families (Logistic Regression, Random Forest, Gradient Boosting) were tuned with "
    "cross-validation on the training data. The held-out test set was used once, for the final evaluation."
)

st.subheader(f"Selected model: {meta['model']}")
cols = st.columns(6)
for col, (label, key) in zip(cols, [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall_sensitivity"), ("Specificity", "specificity"),
                                     ("F1", "f1")]):
    col.metric(label, f"{tm[key]:.2f}")
st.caption("Test-set results for the selected model at the default 0.5 threshold. "
           "The test set has only 60 patients, so each metric has a wide margin of error.")

tab_cmp, tab_eval, tab_thr, tab_imp, tab_method = st.tabs(
    ["Model comparison", "Test-set evaluation", "Threshold trade-off", "What the model relies on", "Method"])

# ------------------------------------------------------------------ comparison
with tab_cmp:
    table = comparison.copy()
    table.insert(1, "Selected", table["Model"].apply(lambda m: "✅" if m in meta["model"] else ""))
    metric_cols = [c for c in table.columns if c not in ("Model", "Selected")]
    try:
        styled = table.style.format({c: "{:.3f}" for c in metric_cols}).highlight_max(
            subset=metric_cols, color="rgba(46,134,171,0.25)")
        st.dataframe(styled, hide_index=True)
    except Exception:
        st.dataframe(table.round(3), hide_index=True)
    st.caption("**CV ROC-AUC**: 5-fold cross-validation on the training data (used to choose the model). "
               "**Test columns**: the 60 held-out patients, used once. The best value in each column is highlighted.")

    long = comparison.melt(id_vars="Model", value_vars=[c for c in comparison.columns if c != "Model"],
                           var_name="Metric", value_name="Score")
    fig = go.Figure()
    palette = ["#2E86AB", "#C0392B", "#7D8CC4", "#F39C12", "#27AE60", "#8E44AD"]
    for color, (metric, grp) in zip(palette, long.groupby("Metric", sort=False)):
        fig.add_bar(x=grp["Model"], y=grp["Score"], name=metric, marker_color=color)
    fig.update_layout(barmode="group", height=420, yaxis_title="Score", yaxis_range=[0, 1.05],
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12))
    H.show_plot(fig, key="cmp_bar")
    st.info(
        "**Read the ranking with care.** The differences between tuned models are small compared with the "
        "fold-to-fold variation in cross-validation (see the repeated-CV check in the notebook), and the test "
        "set is tiny. The final choice therefore rests on the cross-validation score plus practical "
        "considerations, not on a clear win."
    )

# ------------------------------------------------------------------ evaluation
with tab_eval:
    if test_pred is None:
        st.info("`test_predictions.csv` was not found in `models/heart_cleveland/`. Run the extra notebook cell "
                "that saves it to enable the ROC, precision-recall and confusion-matrix views.")
    else:
        y_true = test_pred["y_true"].to_numpy()
        y_score = test_pred["y_score"].to_numpy()
        thr = st.slider("Decision threshold", 0.05, 0.95, default_thr, 0.01,
                        help="Scores at or above this value are flagged as higher estimated risk.")
        st.caption("This slider is for exploring the trade-off on the test patients. The threshold itself "
                   "should be chosen from the cross-validation table in the next tab, not tuned on the test set.")
        pred = (y_score >= thr).astype(int)
        tn = int(((pred == 0) & (y_true == 0)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum()); tp = int(((pred == 1) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        acc = (tp + tn) / len(y_true)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{acc:.2f}")
        k2.metric("Precision", f"{precision:.2f}")
        k3.metric("Recall (disease caught)", f"{recall:.2f}")
        k4.metric("Specificity", f"{spec:.2f}")

        g1, g2, g3 = st.columns(3)
        cm = np.array([[tn, fp], [fn, tp]])
        cm_fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: no disease", "Predicted: disease"], y=["Actual: no disease", "Actual: disease"],
            text=cm, texttemplate="%{text}", textfont={"size": 22}, colorscale="Blues", showscale=False))
        cm_fig.update_layout(title="Confusion matrix", height=360, margin=dict(l=10, r=10, t=50, b=10),
                             yaxis=dict(autorange="reversed"))
        with g1:
            H.show_plot(cm_fig, key="cm")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = float(area_under_curve(fpr, tpr))
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=H.RED, width=3), name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=H.GREY, dash="dash"),
                                     name="Chance"))
        roc_fig.add_trace(go.Scatter(x=[1 - spec], y=[recall], mode="markers", name="Current threshold",
                                     marker=dict(size=12, color="#222")))
        roc_fig.update_layout(title=f"ROC curve (AUC {auc_val:.2f})", height=360, xaxis_title="False positive rate",
                              yaxis_title="True positive rate (recall)", margin=dict(l=10, r=10, t=50, b=10),
                              showlegend=False)
        with g2:
            H.show_plot(roc_fig, key="roc")

        p_arr, r_arr, _ = precision_recall_curve(y_true, y_score)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=r_arr, y=p_arr, mode="lines", line=dict(color=H.BLUE, width=3)))
        if not (np.isnan(precision) or np.isnan(recall)):
            pr_fig.add_trace(go.Scatter(x=[recall], y=[precision], mode="markers",
                                        marker=dict(size=12, color="#222")))
        pr_fig.update_layout(title="Precision-recall curve", height=360, xaxis_title="Recall",
                             yaxis_title="Precision", margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                             yaxis_range=[0, 1.05])
        with g3:
            H.show_plot(pr_fig, key="pr")
        st.caption(f"Based on {len(y_true)} held-out patients ({int(y_true.sum())} with disease). "
                   "In a health-risk setting a missed case (false negative) is usually worse than a false alarm, "
                   "which is why recall is shown next to accuracy.")

# ------------------------------------------------------------------ thresholds
with tab_thr:
    st.markdown(
        "Precision, recall and F1 at different thresholds, computed from **cross-validated predictions on the "
        "training data only**. Lowering the threshold catches more true cases (higher recall) but raises "
        "more false alarms (lower precision)."
    )
    fig = go.Figure()
    for name, color in [("precision", H.BLUE), ("recall", H.RED), ("f1", "#27AE60")]:
        fig.add_trace(go.Scatter(x=thresholds["threshold"], y=thresholds[name], mode="lines+markers", name=name,
                                 line=dict(color=color, width=3)))
    fig.add_vline(x=default_thr, line_dash="dot", line_color=H.GREY, annotation_text="default 0.5")
    fig.update_layout(height=400, xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0, 1.05],
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
    H.show_plot(fig, key="thr")
    with st.expander("Show the table"):
        st.dataframe(thresholds.round(3), hide_index=True)

# ------------------------------------------------------------------ importance
with tab_imp:
    importance = H.global_importance_table()
    top = importance.head(14).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h", marker_color=H.BLUE))
    fig.update_layout(height=480, xaxis_title="Average impact on the score (mean |SHAP|)", yaxis_title=None,
                      margin=dict(l=10, r=10, t=20, b=10))
    H.show_plot(fig, key="imp")
    st.markdown(
        "Features with a larger bar move the model's score more, on average. This describes what the model "
        "**relies on**, not medical cause and effect. The heart-rate-versus-age feature shares its credit with "
        "`age` and `max heart rate`, because they are strongly related."
    )

# ------------------------------------------------------------------ method
with tab_method:
    st.markdown(
        f"""
**Data.** UCI Cleveland heart disease data, 297 patients with complete records and 13 clinical features.
Target: heart disease present (1) or not (0).

**Preprocessing.** Two pipelines, because the model families have different needs:
- Logistic Regression: Yeo-Johnson power transform on continuous features, one-hot encoding with one dropped category.
- Random Forest and Gradient Boosting: standard scaling, full one-hot encoding (trees do not need transformations).

**Feature engineering.** One clinically motivated feature, heart rate relative to the age-expected maximum
(`max heart rate - (220 - age)`). It was tested with and without; the notebook reports how much it changed.

**Tuning and selection.** Randomized search with stratified 5-fold cross-validation on the training set (80%),
scored by ROC-AUC. The final model was selected from the cross-validation results only.

**Evaluation.** A stratified 20% test set (60 patients), used once. Accuracy, precision, recall, specificity,
F1 and ROC-AUC are reported because accuracy alone hides missed cases.

**Explainability.** SHAP values for the selected model, with one-hot columns summed back into their
original feature so each explanation reads like "Chest pain = Asymptomatic".

**Limitations.** Small dataset; one hospital population; the score is not calibrated; the model explains
associations in this data, not causes. Not for clinical use.
"""
    )
