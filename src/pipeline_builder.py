"""
Pipeline builder for credit risk modeling inference.

Builds the preprocessing pipeline shared by training and inference.

Author: Danis Theodoulou
"""

import joblib
from sklearn.pipeline import Pipeline

from transformers import (
    DataSelector,
    MedianImputerTransformer,
    DtypeSetter,
    CategoricalCreator,
    ContinuousTransformer,
    OrdinalEncoderTransformer,
    OneHotEncoderTransformer,
    FeatureEngineeringTransformer
)


def build_preprocessing_pipeline():
    """
    Builds the (unfitted) preprocessing pipeline that turns raw loan data into model features.

    It is fitted on the training split only. The training flow then appends the trained
    model and saves the result as the inference pipeline, so serving runs exactly the
    preprocessing that was fitted during training.

    Returns:
        sklearn Pipeline object
    """
    return Pipeline([
        ('selector', DataSelector()),
        ('imputer', MedianImputerTransformer()),
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('dtype_setter', DtypeSetter()),
        ('categorical_creator', CategoricalCreator()),
        ('continuous_transformer', ContinuousTransformer()),
        ('ordinal_encoder', OrdinalEncoderTransformer()),
        ('onehot_encoder', OneHotEncoderTransformer()),
    ])


def save_pipeline(pipeline, filepath='./models/inference_pipeline.joblib'):
    """
    Saves the inference pipeline to disk.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        filepath: Path to save the pipeline
    """
    joblib.dump(pipeline, filepath)
    print(f"Inference pipeline saved to {filepath}")


def load_pipeline(filepath='./models/inference_pipeline.joblib'):
    """
    Loads a saved inference pipeline from disk.
    
    Args:
        filepath: Path to the saved pipeline
    
    Returns:
        sklearn Pipeline object
    """
    pipeline = joblib.load(filepath)
    print(f"Inference pipeline loaded from {filepath}")
    return pipeline
