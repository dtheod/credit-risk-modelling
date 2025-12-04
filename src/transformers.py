"""
Custom sklearn transformers for credit risk modeling inference pipeline.

These transformers reuse existing functions from data_processing.py and feature_engineering.py
to ensure consistency between training and inference. Only artifact-loading transformers
(KNN imputer, encoders) have custom logic.

Author: Danis Theodoulou
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Import existing functions to reuse logic
from data_processing import (
    categorical_variables_task,
    continuous_variables_transformations,
    set_column_dtypes,
    clean_data
)
from feature_engineering import (
    engineered_features
)
from data_load import select_columns


class DataSelector(BaseEstimator, TransformerMixin):
    """Selects relevant columns from the raw data using existing select_columns function."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Reuse the existing select_columns function
        return select_columns(X)


class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    """Loads pre-fitted KNN imputer and applies it to continuous variables."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        imputer = joblib.load("./models/knn_imputer.joblib")
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X.copy()
        X[continuous_cols] = imputer.transform(X[continuous_cols])
        return X


class DtypeSetter(BaseEstimator, TransformerMixin):
    """Converts float columns to integers using existing set_column_dtypes function."""
    
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
    """Applies log transformations using existing continuous_variables_transformations function."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Reuse the existing continuous_variables_transformations function
        return continuous_variables_transformations(X)

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
    """Loads pre-fitted ordinal encoder for CreditScoreEsMicroL and Rating."""
    
    def __init__(self, encoder_path='./models/ordinal_encoder.joblib'):
        self.encoder = joblib.load(encoder_path)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Check if both columns exist
        if 'CreditScoreEsMicroL' in X.columns:
            # Encode both columns at once (same as training)
            encoded_values = self.encoder.transform(X[['CreditScoreEsMicroL']])
            X['CreditScoreEsMicroLEnc'] = encoded_values[:, 0]
            X = X.drop(['CreditScoreEsMicroL'], axis=1)
        
        return X


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """Loads pre-fitted one-hot encoder for categorical variables."""
    
    def __init__(self, encoder_path='./models/one_hot_encoder.joblib'):
        self.encoder = joblib.load(encoder_path)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X.drop(columns=["LoanId"], inplace=True)
        categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        non_categorical_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()
        
        
        if len(categorical_cols) > 0:
            # Convert to string (same as training)
            categorical_features = X[categorical_cols].astype(str)
            encoded_features = self.encoder.transform(categorical_features)
            encoded_df = pd.DataFrame(
                encoded_features.toarray(),
                columns=self.encoder.get_feature_names_out(),
                index=X.index
            )
            
            non_categorical_df = X[non_categorical_cols]
            result = pd.concat([non_categorical_df.reset_index(drop=True),
                               encoded_df.reset_index(drop=True)], axis=1)
        else:
            return X
        
        return result

class XGBoostingTransformer(BaseEstimator, TransformerMixin):
    """Loads pre-fitted XGBoost model and applies it to the data."""
    
    def __init__(self, model_path='./models/xgboost_model.joblib'):
        self.model = joblib.load(model_path)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        predictions = self.model.predict_proba(X)[:, 1]
        return predictions
