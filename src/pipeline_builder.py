"""
Pipeline builder for credit risk modeling inference.

Builds inference pipeline that loads pre-fitted artifacts (encoders, imputers, model).

Author: Danis Theodoulou
"""

import joblib
from sklearn.pipeline import Pipeline

from transformers import (
    DataSelector,
    KNNImputerTransformer,
    DtypeSetter,
    CategoricalCreator,
    ContinuousTransformer,
    OrdinalEncoderTransformer,
    OneHotEncoderTransformer,
    XGBoostingTransformer,
    FeatureEngineeringTransformer
)


def build_inference_pipeline(
    model_path,
    knn_imputer_path,
    ordinal_encoder_path,
    onehot_encoder_path
):
    """
    Builds the inference pipeline that loads all pre-fitted artifacts.
    
    This pipeline is used for making predictions on new data using the
    artifacts saved during training.
    
    Args:
        model_path: Path to saved XGBoost model
        knn_imputer_path: Path to saved KNN imputer
        ordinal_encoder_path: Path to saved ordinal encoder
        onehot_encoder_path: Path to saved one-hot encoder
    
    Returns:
        sklearn Pipeline object for inference
    """
    
    # Build inference pipeline
    inference_pipeline = Pipeline([
        ('selector', DataSelector()),
        ('knn_imputer', KNNImputerTransformer()),
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('dtype_setter', DtypeSetter()),
        ('categorical_creator', CategoricalCreator()),
        ('continuous_transformer', ContinuousTransformer()),
        ('ordinal_encoder', OrdinalEncoderTransformer(encoder_path=ordinal_encoder_path)),
        ('onehot_encoder', OneHotEncoderTransformer(encoder_path=onehot_encoder_path)),
        ('model', XGBoostingTransformer(model_path=model_path))
    ])
    
    return inference_pipeline


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
