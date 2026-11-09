# ❤️ HeartRisk AI

**Heart disease risk estimation with explainable machine learning.** An interactive Streamlit app that takes a patient's clinical measurements, returns a model-estimated risk score, and explains *why* with SHAP.

**Live demo:** _add your Streamlit Community Cloud link here_

> ⚠️ **Educational project, not a medical device.** The score is an estimate from a model trained on a small public dataset (297 patients). It is not a diagnosis and must not be used for medical decisions.

_Add 2-3 screenshots here (Risk Prediction, Model Comparison, Data Insights)._

## Features

- **Risk Prediction:** plain-language form, risk gauge, per-patient SHAP explanation, what-if explorer, downloadable summary, warnings for inputs outside the training range
- **Model Comparison:** cross-validated and test-set metrics for three tuned models, confusion matrix, ROC and precision-recall curves, threshold trade-off, global feature importance
- **Data Insights:** interactive exploration of the dataset with filters, distributions, category-level disease rates and correlations

## How the model was built

| Step | What was done |
|---|---|
| Data | UCI Cleveland heart disease data, 297 complete records, 13 clinical features |
| EDA | Data-quality checks, distributions by outcome, category-level disease rates, correlation analysis |
| Preprocessing | Two scikit-learn pipelines: power transform + one-hot (drop first) for Logistic Regression; scaling + one-hot for tree models |
| Feature engineering | Heart rate relative to the age-expected maximum, tested with and without |
| Models | Logistic Regression, Random Forest, Gradient Boosting (XGBoost) |
| Tuning | Randomized search, stratified 5-fold cross-validation, ROC-AUC |
| Selection | From cross-validation only; repeated CV used to check how separable the models are |
| Evaluation | Stratified held-out test set used once: accuracy, precision, recall, specificity, F1, ROC-AUC |
| Explainability | SHAP, with one-hot columns summed back into their original feature |

**Results:** _fill in from the Model Comparison page, for example "test ROC-AUC 0.94, recall 0.71 on 60 held-out patients"._

## Project structure

```
heart-risk-ai/
├── app.py                      # entry point and navigation
├── views/                      # the four pages
│   ├── home.py
│   ├── risk_prediction.py
│   ├── model_comparison.py
│   └── data_insights.py
├── utils/
│   ├── app_helpers.py          # paths, cached loaders, styling
│   ├── explainability.py       # SHAP helpers (written by the notebook)
│   └── preprocessing.py        # custom transformer (written by the notebook)
├── models/heart_cleveland/     # saved pipeline, metadata, evaluation files
├── data/heart_cleveland_clean.csv
├── notebooks/heart_disease_analysis_final.ipynb
├── tools/make_requirements.py
├── .streamlit/config.toml
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The `models/` folder is produced by running the notebook top to bottom. The saved model is a pickle, so the scikit-learn version in `requirements.txt` must match the one used for training (`python tools/make_requirements.py` pins your current versions).

## Limitations

- Small, single-source dataset; results would change with different data
- The risk score is not a calibrated probability
- SHAP explains the model's behaviour, not medical cause and effect
- The test set has only 60 patients, so performance numbers are rough estimates

## Data

UCI Machine Learning Repository, Heart Disease dataset (Cleveland database). Investigators: Janosi, Steinbrunn, Pfisterer, Detrano. https://archive.ics.uci.edu/dataset/45/heart+disease

## Author

_Your name · link to LinkedIn / GitHub_
