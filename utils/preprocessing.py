import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class addheartfeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_age_thalach: bool = True):
        self.add_age_thalach = add_age_thalach

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_age_thalach:
            X["age_thalach"] = X["thalach"] - (220 - X["age"])
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if self.add_age_thalach:
            names.append("age_thalach")
        return np.asarray(names, dtype=object)
