"""HeartRisk AI - Streamlit entry point. Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from utils import app_helpers as H

st.set_page_config(page_title="HeartRisk AI", page_icon="❤️", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/risk_prediction.py", title="Risk Prediction", icon="❤️"),
    st.Page("views/model_comparison.py", title="Model Comparison", icon="📊"),
    st.Page("views/data_insights.py", title="Data Insights", icon="🔍"),
]
navigation = st.navigation(pages)

H.inject_css()
if not H.missing_files():
    H.sidebar_info()

navigation.run()
