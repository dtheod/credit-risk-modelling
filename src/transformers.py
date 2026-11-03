"""
Custom sklearn transformers for credit risk modeling.

These transformers reuse existing functions from data_processing.py and feature_engineering.py
so that training and inference share the same code. Transformers with state (imputer, log caps,
encoders) learn it in `fit`, so fitting the pipeline on the training split is the only place
that state is estimated.

Author: Danis Theodoulou
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Import existing functions to reuse logic
from data_processing import (
    LOG_TRANSFORM_COLUMNS,
    categorical_variables_task,
    continuous_variables_transformations,
    set_column_dtypes,
)
from feature_engineering import (
    engineered_features
)
from data_load import select_columns


class DataSelector(BaseEstimator, TransformerMixin):
    """Selects relevant columns from the raw data using existing select_columns function."""

    def fit(self, X, y=None):
        self.numeric_cols_ = select_columns(X).select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X, y=None):
        # Single API rows with missing values arrive as object columns
        X = X.copy()
        X[self.numeric_cols_] = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        # Reuse the existing select_columns function
        return select_columns(X)


class MedianImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputes numeric columns with the median learned on the training data."""

    def fit(self, X, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(X[self.numeric_cols_])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.numeric_cols_] = self.imputer_.transform(X[self.numeric_cols_])
        return X


class DtypeSetter(BaseEstimator, TransformerMixin):
    """Rounds code/count columns to integers using existing set_column_dtypes function."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Reuse the existing set_column_dtypes function
        return set_column_dtypes(X)


class CategoricalCreator(BaseEstimator, TransformerMixin):
    """Creates categorical variables using existing categorical_variables_task function."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Reuse the existing categorical_variables_task function
        return categorical_variables_task(X)


class ContinuousTransformer(BaseEstimator, TransformerMixin):
    """Log-transforms skewed columns, clipping at the 99.5th percentile learned on the training data."""

    def fit(self, X, y=None):
        self.caps_ = {col: X[col].quantile(0.995) for col in LOG_TRANSFORM_COLUMNS}
        return self

    def transform(self, X, y=None):
        # Reuse the existing continuous_variables_transformations function
        return continuous_variables_transformations(X.copy(), self.caps_)

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Applies feature engineering using existing engineered_features function."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Reuse the existing engineered_features function
        return engineered_features(X)


class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Ordinal-encodes CreditScoreEsMicroL (M1-M10, subprime categories)."""

    def fit(self, X, y=None):
        subprime_categories = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
        self.encoder_ = OrdinalEncoder(
            categories=[subprime_categories],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.encoder_.fit(X[["CreditScoreEsMicroL"]])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["CreditScoreEsMicroLEnc"] = self.encoder_.transform(X[["CreditScoreEsMicroL"]])[:, 0]
        return X.drop(["CreditScoreEsMicroL"], axis=1)


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """One-hot encodes categorical columns, with categories learned on the training data."""

    def fit(self, X, y=None):
        X = X.drop(columns=["LoanId"], errors="ignore")
        self.categorical_cols_ = X.select_dtypes(include=["category", "object"]).columns.tolist()
        self.non_categorical_cols_ = X.select_dtypes(exclude=["category", "object"]).columns.tolist()
        self.encoder_ = OneHotEncoder(handle_unknown="ignore")
        # Convert to string to handle mixed types
        self.encoder_.fit(X[self.categorical_cols_].astype(str))
        return self

    def transform(self, X, y=None):
        encoded_df = pd.DataFrame(
            self.encoder_.transform(X[self.categorical_cols_].astype(str)).toarray(),
            columns=self.encoder_.get_feature_names_out(),
        )
        return pd.concat([X[self.non_categorical_cols_].reset_index(drop=True), encoded_df], axis=1)
