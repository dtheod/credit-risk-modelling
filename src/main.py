"""
main.py file that you can call all pipelines form both models.
Example:
uv run src/main.py
"""

from prefect import flow
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from data_load import (
    select_columns,
    data_preprocessing_lgd
)
from data_processing import (
    create_pd12_target,
    categorical_variables_task,
    continuous_variables_transformations,
    knn_impute_continuous_variables,
    set_column_dtypes,
    ordinal_encoding,
    clean_data,
    ordinal_encoding_lgd
)
from feature_engineering import (
    train_test_split_task,
    one_hot_encode_features,
    engineered_features,
    feature_engineering_lgd
)
from train_model import (
    train_xgboost_model,
    train_regression_model
)
import pandas as pd
import hydra
import os
from omegaconf import DictConfig
from datetime import datetime
import joblib

from pipeline_builder import (
    build_inference_pipeline,
    save_pipeline,
    load_pipeline
)
import xgboost as xgb
import matplotlib.pyplot as pl


DATE = datetime.today().strftime("%Y-%m-%d %H:%M")

@flow(flow_run_name=f"ML training pipeline run on {DATE}", log_prints=True)
def pd_training_evaluation_pipeline(config: DictConfig) -> pd.DataFrame:

    # Setting up mlflow for tracking
    # Using sqlite as tracking uri for simplicity. 
    # The db should be visible in the root directory after the pipeline is run.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pd_model_experiment")

    # Creating the target variable and saving the processed data for prediction 
    # using raw form and using pipeline
    data_processed = (
        pd.read_csv(config.data.raw_data, sep = ";")
        .pipe(clean_data)
        .pipe(create_pd12_target)    
    )
    data_processed.to_csv(config.project.data.processed, index=False)

    # Processing the data for training(Main functions in pipeline)
    data_features = (
        data_processed
        .pipe(select_columns)
        .pipe(knn_impute_continuous_variables)
        .pipe(engineered_features)
        .pipe(set_column_dtypes)
        .pipe(categorical_variables_task)
        .pipe(continuous_variables_transformations)
        .pipe(ordinal_encoding)
        .pipe(one_hot_encode_features)
        .merge(
            data_processed.filter(["LoanId", "DefaultTarget_12m"]),
            on = "LoanId",
            how = "inner"
        )
        .astype({"DefaultTarget_12m": "int64"})
        .drop(columns=["LoanId"])
    )
    
    data_features.to_csv(config.project.data.features, index=False)

    # Splitting the data into train and test using stratified sampling to maintain class balance
    train_data, test_data, train_target, test_target = train_test_split_task(
        data_features.drop(columns=["DefaultTarget_12m"]),
        data_features["DefaultTarget_12m"]
    )
    # Saving data for tracalbilty and evaluation
    pd.concat([train_data, train_target], axis=1).to_csv(config.project.data.train, index=False)
    pd.concat([test_data, test_target], axis=1).to_csv(config.project.data.test, index=False)
    
    # Runinng Optuna for training and hyperparameter optimization
    model = train_xgboost_model(
        train_data, 
        train_target
    )
    
    # Save the trained model
    joblib.dump(model, config.artifacts.xgboost_model)
    print(f"Model saved to {config.artifacts.xgboost_model}")
    
    return None


@flow(flow_run_name=f"ML Inference pipeline run on {DATE}", log_prints=True)
def pd_inference_pipeline(config: DictConfig) -> pd.DataFrame:
    """
    This will task the test set and the final predictions, interpretability and evaluate the model 
    """
    pipeline = build_inference_pipeline(
        model_path=config.artifacts.xgboost_model,
        knn_imputer_path=config.artifacts.knn_imputer,
        ordinal_encoder_path=config.artifacts.ordinal_encoder,
        onehot_encoder_path=config.artifacts.one_hot_encoder,
    )
    save_pipeline(pipeline)

    #Testing the pipeline
    data = (
        pd.read_csv(config.project.data.processed)
        .head(1)
    )
    results = pipeline.transform(data)
    
    return None

@flow(flow_run_name= f"LGD training pipeline run on {DATE}", log_prints=True)
def lgd_training_evaluation_pipeline(config: DictConfig) -> pd.DataFrame:

    features = (
        pd.read_csv(config.data.raw_data, sep = ";")
        .pipe(data_preprocessing_lgd)
        .pipe(feature_engineering_lgd)
        .pipe(ordinal_encoding_lgd)
    )
    #Saving the features for tracking
    features.to_csv(config.project.data.features, index=False)


    lgd = features["LossGivenDefault"]
    X_train, X_test, y_train, y_test = train_test_split(
        features.drop("LossGivenDefault", axis=1),
        lgd,
        test_size=0.1,
        random_state=42
    )
    pd.concat([X_train, y_train], axis=1).to_csv(config.project.data.train, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(config.project.data.test, index=False)
    model = train_regression_model(X_train, y_train)
    joblib.dump(model, config.artifacts.catboost_regression_model)
    
    return None

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:

    if config.project.name == "lgd_model":
        if config.project.pipeline == "training":
            lgd_training_evaluation_pipeline(config)
        else:
            raise ValueError(f"Pipeline {config.project.pipeline} not supported")

    elif config.project.name == "pd_model":
        if config.project.pipeline == "training":
            pd_training_evaluation_pipeline(config)
        elif config.project.pipeline == "inference":
            pd_inference_pipeline(config)
        else:
            raise ValueError(f"Pipeline {config.project.pipeline} not supported")
    else:
        raise ValueError(f"Project {config.project} not supported")

if __name__ == "__main__":
    main()
