import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PRETTY = {
    "age": "Age",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "thalach": "Max heart rate",
    "oldpeak": "ST depression",
    "age_thalach": "Heart rate vs age-expected max",
    "sex": "Sex (male)",
    "fbs": "Fasting blood sugar > 120",
    "exang": "Exercise-induced angina",
    "cp": "Chest pain",
    "restecg": "Resting ECG",
    "slope": "ST slope",
    "ca": "Major vessels",
    "thal": "Thal test",
}
TREE_MODELS = ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier")


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def readable_name(raw_name, labels):
    """'cat__cp_3' -> 'Chest pain: Asymptomatic'; 'continous__oldpeak' -> 'ST depression'."""
    base = raw_name.split("__", 1)[1]
    if base in PRETTY:
        return PRETTY[base]
    col, code = base.rsplit("_", 1)
    code = str(int(float(code)))
    return PRETTY.get(col, col) + ": " + labels.get(col, {}).get(code, code)


def _group_of(raw_name):
    """'cat__cp_3' -> 'cp'; 'continous__oldpeak' -> 'oldpeak'."""
    base = raw_name.split("__", 1)[1]
    return base if base in PRETTY else base.rsplit("_", 1)[0]


def class1_values(explainer, X):
    """SHAP values for class 1 (disease), shape (n_rows, n_features)."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def load_artifacts(artifact_dir):
    model = joblib.load(os.path.join(artifact_dir, "heart_risk_pipeline.joblib"))
    with open(os.path.join(artifact_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    X_train = pd.read_csv(os.path.join(artifact_dir, "X_train.csv"))[meta["feature_columns"]]
    pre, clf = model[:-1], model[-1]
    background = _dense(pre.transform(X_train))
    if type(clf).__name__ in TREE_MODELS:
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, background)
    labels = meta["category_labels"]
    raw_names = list(pre.get_feature_names_out())
    names = [readable_name(n, labels) for n in raw_names]
    groups = [_group_of(n) for n in raw_names]
    orig_features = list(dict.fromkeys(groups))
    G = np.array([[1.0 if g == o else 0.0 for o in orig_features] for g in groups])  # one-hot column -> original feature
    scale = "probability" if type(clf).__name__ == "RandomForestClassifier" else "log-odds"
    return {"model": model, "pre": pre, "clf": clf, "meta": meta, "X_train": X_train,
            "background": background, "explainer": explainer, "names": names, "scale": scale,
            "orig_features": orig_features, "G": G}


def global_importance(art):
    """Mean |SHAP| per ORIGINAL feature (one-hot columns are summed back together first)."""
    sv = class1_values(art["explainer"], art["background"])
    agg = sv @ art["G"]
    return (pd.DataFrame({"feature": [PRETTY[o] for o in art["orig_features"]],
                          "mean_abs_shap": np.abs(agg).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def plot_global(art, max_display=15):
    """Returns (bar figure per original feature, beeswarm figure per encoded column)."""
    imp = global_importance(art).head(max_display).iloc[::-1]
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Average impact on the model's score (mean |SHAP|)")
    ax.set_title("Which features matter most overall")
    fig_bar.tight_layout()
    sv = class1_values(art["explainer"], art["background"])
    plt.figure()
    shap.summary_plot(sv, art["background"], feature_names=art["names"],
                      max_display=max_display, show=False)
    plt.title("Each dot is a patient (red = high value, or category present)")
    fig_swarm = plt.gcf(); fig_swarm.tight_layout()
    return fig_bar, fig_swarm


def explain_patient(art, patient):
    """patient: dict with the 13 raw columns. Contributions are per original feature, labelled with the patient's value."""
    row = pd.DataFrame([patient])[art["meta"]["feature_columns"]]
    risk = float(art["model"].predict_proba(row)[0, 1])
    engineered = art["model"][0].transform(row)          # adds age_thalach when the model uses it
    x = _dense(art["pre"].transform(row))
    sv = class1_values(art["explainer"], x)[0]
    agg = sv @ art["G"]
    labels = art["meta"]["category_labels"]
    shown = []
    for o in art["orig_features"]:
        v = engineered[o].iloc[0]
        text = labels[o].get(str(int(v)), str(int(v))) if o in labels else f"{v:g}"
        shown.append(f"{PRETTY[o]} = {text}")
    base = float(np.ravel(art["explainer"].expected_value)[-1])
    table = (pd.DataFrame({"feature": shown, "shap": agg})
             .assign(abs_shap=lambda d: d["shap"].abs())
             .sort_values("abs_shap", ascending=False)
             .drop(columns="abs_shap").reset_index(drop=True))
    return {"risk": risk, "base": base, "agg": agg, "names": shown, "table": table}


def check_additivity(art, result):
    """base + sum(SHAP) must equal the model output (probability for Random Forest, log-odds otherwise)."""
    lhs = result["base"] + float(result["agg"].sum())
    rhs = result["risk"] if art["scale"] == "probability" else float(np.log(result["risk"] / (1 - result["risk"])))
    return lhs, rhs


def explain_sentence(result, top_n=3):
    t = result["table"]
    up = t[t["shap"] > 0].head(top_n)["feature"].tolist()
    down = t[t["shap"] < 0].head(top_n)["feature"].tolist()
    text = f"Model-estimated risk score: {result['risk']:.0%}."
    if up:
        text += " Factors that pushed the score up: " + ", ".join(up) + "."
    if down:
        text += " Factors that pushed it down: " + ", ".join(down) + "."
    return text


def plot_patient(art, result, max_display=10):
    """Waterfall plot for one patient. Returns the figure."""
    expl = shap.Explanation(values=result["agg"], base_values=result["base"],
                            feature_names=result["names"])
    plt.figure()
    shap.plots.waterfall(expl, max_display=max_display, show=False)
    return plt.gcf()
