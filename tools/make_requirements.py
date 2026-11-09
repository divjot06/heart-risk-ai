"""Write requirements.txt pinned to the versions in THIS environment.

Run it from the project root inside the same conda environment you used for the notebook:
    python tools/make_requirements.py

The saved model is a pickle, so Streamlit Community Cloud must install the same scikit-learn
(and numpy / pandas) versions that trained it. This script copies them from your environment.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

packages = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "scipy", "joblib", "matplotlib", "shap"]

# xgboost is only needed if the saved model uses it
try:
    import joblib
    model = joblib.load(ROOT / "models" / "heart_cleveland" / "heart_risk_pipeline.joblib")
    if "xgboost" in type(model[-1]).__module__:
        packages.append("xgboost")
except Exception as exc:  # keep going, but tell the user
    print("Could not inspect the saved model (is models/heart_cleveland/ in place?):", exc)

lines, missing = [], []
for pkg in packages:
    try:
        lines.append(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        missing.append(pkg)

if missing:
    print("Not installed in this environment:", ", ".join(missing))
    print("Install them with:  pip install " + " ".join(missing))
    sys.exit(1)

(ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Wrote requirements.txt:")
print("\n".join(lines))
print("\nPython version:", sys.version.split()[0], "-> choose the same minor version on Streamlit Community Cloud.")
