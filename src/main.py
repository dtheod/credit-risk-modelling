"""
main.py file that you can call all pipelines form both models.
Example:
uv run src/main.py
"""

from prefect import flow
import mlflow
from data_load import data_preprocessing_lgd
from data_processing import (
    create_pd12_target,
    clean_data,
)
from feature_engineering import (
    out_of_time_split,
    feature_engineering_lgd,
    LGD_CATEGORICAL_FEATURES,
)
from train_model import (
    train_xgboost_model,
    train_regression_model
)
from evaluation import evaluate_pd_model, evaluate_lgd_model
from interpretability import plot_feature_importance, shap_summary_plot
import pandas as pd
import hydra
import os
from omegaconf import DictConfig
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline

from pipeline_builder import (
    build_preprocessing_pipeline,
    save_pipeline,
    load_pipeline
)


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

    # Out-of-time split on the raw rows, before any preprocessing is fitted
    train_raw, test_raw = out_of_time_split(data_processed, "LoanDate")

    # Preprocessing (imputer, log caps, encoders) is fitted on the training split only
    preprocessor = build_preprocessing_pipeline()
    train_data = preprocessor.fit_transform(train_raw)
    test_data = preprocessor.transform(test_raw)
    train_target = train_raw["DefaultTarget_12m"].reset_index(drop=True)
    test_target = test_raw["DefaultTarget_12m"].reset_index(drop=True)

    # Saving data for tracalbilty and evaluation
    pd.concat([train_data, train_target], axis=1).to_csv(config.project.data.train, index=False)
    pd.concat([test_data, test_target], axis=1).to_csv(config.project.data.test, index=False)

    with mlflow.start_run(run_name="pd_model_run", tags={"model": "xgboost"}):
        # Runinng Optuna for training and hyperparameter optimization
        model = train_xgboost_model(
            train_data,
            train_target
        )

        evaluate_pd_model(
            test_target,
            model.predict_proba(test_data)[:, 1],
            test_raw["Country"].reset_index(drop=True),
            metrics_path=config.artifacts.pd_metrics,
            plot_path=config.artifacts.pd_evaluation_plot,
        )

    plot_feature_importance(model)
    shap_summary_plot(model, test_data)

    # Save the trained model, and the fitted preprocessing + model as the inference pipeline
    joblib.dump(model, config.artifacts.xgboost_model)
    print(f"Model saved to {config.artifacts.xgboost_model}")
    save_pipeline(Pipeline(preprocessor.steps + [("model", model)]), config.artifacts.inference_pipeline)

    return None


@flow(flow_run_name=f"ML Inference pipeline run on {DATE}", log_prints=True)
def pd_inference_pipeline(config: DictConfig) -> pd.DataFrame:
    """
    Smoke-tests the saved inference pipeline on raw rows from the processed data.
    """
    pipeline = load_pipeline(config.artifacts.inference_pipeline)
    data = pd.read_csv(config.project.data.processed).head(5)
    print(f"Predicted PDs: {pipeline.predict_proba(data)[:, 1]}")

    return None

@flow(flow_run_name= f"LGD training pipeline run on {DATE}", log_prints=True)
def lgd_training_evaluation_pipeline(config: DictConfig) -> pd.DataFrame:

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lgd_model_experiment")

    features = (
        pd.read_csv(config.data.raw_data, sep = ";")
        .pipe(data_preprocessing_lgd)
        .pipe(feature_engineering_lgd)
    )
    #Saving the features for tracking
    features.to_csv(config.project.data.features, index=False)

    # Out-of-time split on default date
    train, test = out_of_time_split(features, "DefaultDate")
    X_train, y_train = train.drop(columns=["LGD", "DefaultDate"]), train["LGD"]
    X_test, y_test = test.drop(columns=["LGD", "DefaultDate"]), test["LGD"]
    pd.concat([X_train, y_train], axis=1).to_csv(config.project.data.train, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(config.project.data.test, index=False)

    with mlflow.start_run(run_name="lgd_model_run", tags={"model": "catboost"}):
        model = train_regression_model(X_train, y_train, LGD_CATEGORICAL_FEATURES)
        evaluate_lgd_model(
            y_test,
            model.predict(X_test).clip(0, 1),
            y_train.mean(),
            metrics_path=config.artifacts.lgd_metrics,
        )
    joblib.dump(model, config.artifacts.catboost_regression_model)
    
    return None

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:

    for path in config.project.data.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

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
