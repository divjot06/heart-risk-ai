import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import app_helpers as H

H.require_artifacts()
meta = H.load_meta()
labels = meta["category_labels"]
raw = H.load_data()

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
PRETTY = H.PRETTY_FEATURES

st.title("🔍 Data Insights")
st.markdown("Explore the dataset behind the model: 297 patients from the UCI Cleveland heart disease data. "
            "Use the filters in the sidebar to focus on a group.")

# ------------------------------------------------------------------ filters
with st.sidebar:
    st.subheader("Filters")
    sex_codes = st.multiselect("Sex", [0, 1], default=[0, 1], format_func=lambda c: labels["sex"][str(c)])
    age_min, age_max = int(raw["age"].min()), int(raw["age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

df = raw[raw["sex"].isin(sex_codes) & raw["age"].between(*age_range)].copy()
if df.empty:
    st.warning("No patients match these filters. Widen the filters in the sidebar.")
    st.stop()
df["Outcome"] = df["target"].map(H.TARGET_NAMES)

tab_over, tab_dist, tab_cat, tab_corr, tab_data = st.tabs(
    ["Overview", "Distributions", "Categories", "Correlations", "Data"])

# ------------------------------------------------------------------ overview
with tab_over:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients (filtered)", f"{len(df)}")
    m2.metric("With heart disease", f"{df['target'].mean():.0%}")
    m3.metric("Median age", f"{df['age'].median():.0f}")
    m4.metric("Male", f"{df['sex'].mean():.0%}")

    a, b = st.columns(2)
    counts = df["Outcome"].value_counts().reindex(["No disease", "Disease"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[H.BLUE, H.RED],
                           text=counts.values.astype(int), textposition="outside"))
    fig.update_layout(title="Class balance", height=340, yaxis_title="Patients", margin=dict(l=10, r=10, t=50, b=10))
    with a:
        H.show_plot(fig, key="balance")
    fig = px.histogram(df, x="age", color="Outcome", barmode="overlay", opacity=0.65, nbins=16,
                       color_discrete_map=H.TARGET_COLORS, title="Age by outcome")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    with b:
        H.show_plot(fig, key="age_hist")

    corr_t = df.drop(columns="Outcome").corr()["target"].drop("target")
    strongest = corr_t.reindex(corr_t.abs().sort_values(ascending=False).index).head(3)
    st.markdown("**Strongest linear links with heart disease in this selection:** " +
                ", ".join(f"{PRETTY[k]} ({v:+.2f})" for k, v in strongest.items()) + ".")
    st.caption("Class balance is close to even, so no resampling was needed.")

# ------------------------------------------------------------------ distributions
with tab_dist:
    feature = st.selectbox("Measurement", NUMERIC, format_func=lambda c: PRETTY[c], key="dist_feature")
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x=feature, color="Outcome", barmode="overlay", opacity=0.65, nbins=20,
                       color_discrete_map=H.TARGET_COLORS, title=f"{PRETTY[feature]}: distribution")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    with c1:
        H.show_plot(fig, key="dist_hist")
    fig = px.box(df, x="Outcome", y=feature, color="Outcome", color_discrete_map=H.TARGET_COLORS, points="outliers",
                 category_orders={"Outcome": ["No disease", "Disease"]}, title=f"{PRETTY[feature]} by outcome")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    with c2:
        H.show_plot(fig, key="dist_box")
    med = df.groupby("Outcome")[feature].median()
    if {"No disease", "Disease"} <= set(med.index):
        st.markdown(f"Median **{PRETTY[feature].lower()}**: {med['No disease']:.1f} without disease, "
                    f"{med['Disease']:.1f} with disease.")

# ------------------------------------------------------------------ categories
with tab_cat:
    feature = st.selectbox("Feature", CATEGORICAL, format_func=lambda c: PRETTY[c], key="cat_feature")
    grouped = df.groupby(feature)["target"].agg(rate="mean", n="size").reset_index()
    grouped["label"] = grouped[feature].apply(lambda v: labels[feature].get(str(int(v)), str(int(v))))
    small = grouped["n"] < 20
    fig = go.Figure(go.Bar(
        x=grouped["label"], y=grouped["rate"] * 100,
        marker_color=[H.GREY if s else H.RED for s in small],
        text=[f"{r:.0%}<br>n={n}" for r, n in zip(grouped["rate"], grouped["n"])], textposition="outside"))
    fig.update_layout(title=f"Share of patients with heart disease, by {PRETTY[feature].lower()}", height=420,
                      yaxis_title="% with disease", yaxis_range=[0, 115], margin=dict(l=10, r=10, t=50, b=10))
    H.show_plot(fig, key="cat_bar")
    note = "Grey bars have fewer than 20 patients, so their rates are unreliable." if small.any() else ""
    st.caption("Each bar is the percentage of patients in that category who have heart disease; "
               "n is the number of patients. " + note)

# ------------------------------------------------------------------ correlations
with tab_corr:
    corr = df.drop(columns="Outcome").corr()
    names = [PRETTY[c] for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=corr.round(2).values, texttemplate="%{text}", textfont={"size": 9}))
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(autorange="reversed"))
    H.show_plot(fig, key="corr_heat")
    tgt = corr["target"].drop("target").sort_values()
    fig = go.Figure(go.Bar(x=tgt.values, y=[PRETTY[c] for c in tgt.index], orientation="h",
                           marker_color=[H.RED if v > 0 else H.BLUE for v in tgt.values]))
    fig.update_layout(title="Correlation with heart disease", height=460, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Pearson correlation")
    H.show_plot(fig, key="corr_bar")
    st.caption("Pearson correlation treats coded categories as numbers, so for multi-category features such as "
               "chest pain it is only a rough guide. The Categories tab is more reliable for those.")

# ------------------------------------------------------------------ data
with tab_data:
    shown = df.drop(columns=["Outcome"]).copy()
    for col in CATEGORICAL:
        shown[col] = shown[col].apply(lambda v, col=col: labels[col].get(str(int(v)), str(int(v))))
    shown["target"] = shown["target"].map(H.TARGET_NAMES)
    shown = shown.rename(columns=PRETTY)
    st.dataframe(shown, hide_index=True)
    st.download_button("Download filtered data (CSV)", shown.to_csv(index=False), file_name="heart_filtered.csv",
                       mime="text/csv")
    st.caption("Source: UCI Machine Learning Repository, Cleveland heart disease database (297 complete records).")
