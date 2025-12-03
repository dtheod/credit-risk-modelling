"""
Training Script for PD model using XGBoost
Author: Danis Theodoulou
"""

from xgboost import XGBClassifier
import mlflow
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_parallel_coordinate, plot_slice
import os
from prefect import task

@task
def train_xgboost_model(X_train, y_train, n_trials=20):
    
    # Stratified K-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define Optuna objective function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'enable_categorical': True,
            'random_state': 42,
            'objective': 'binary:logistic'
        }
        
        model = XGBClassifier(**params)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return np.mean(cv_scores)
    
    # Run Optuna optimization
    print(f"Optuna hyperparameters optimization with {n_trials} trials...")
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save Optuna plots
    save_optuna_plots(study, "pd_model")
    
    # Print best results
    print(f"\nBest trial:")
    print(f"  ROC AUC: {study.best_value:.4f}")
    print(f"  Best hyperparameters: {study.best_params}")
    
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    best_params['enable_categorical'] = True
    best_params['random_state'] = 42
    
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    with mlflow.start_run(run_name="pd_model_run", tags={"model": "xgboost"}):
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", study.best_value)
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature names
        mlflow.log_param("features", X_train.columns.tolist())
    
    return model

@task
def train_regression_model(X_train, y_train, n_trials=20):
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'MAE',
            'verbose': False,
            'random_state': 42
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        model = CatBoostRegressor(**params)
        
        # cross_val_score returns negative RMSE, so we negate it to get positive RMSE
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial MAE: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    
    save_optuna_plots(study, "lgd_model")
    
    # Train final model with best params
    best_params = study.best_params
    best_params['loss_function'] = 'RMSE'
    best_params['verbose'] = False
    best_params['random_state'] = 42
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    return final_model

    


@task
def save_optuna_plots(study, model_type):
    # Parameter importance plot
    fig_importance = plot_param_importances(study)
    fig_importance.write_image(f"./models/optuna_plots/{model_type}/param_importance.png")
    print("  Saved: param_importance.png")
    
    # Optimization history plot
    fig_history = plot_optimization_history(study)
    fig_history.write_image(f"./models/optuna_plots/{model_type}/optimization_history.png")
    print("  Saved: optimization_history.png")
    
    # Parallel coordinate plot
    fig_parallel = plot_parallel_coordinate(study)
    fig_parallel.write_image(f"./models/optuna_plots/{model_type}/parallel_coordinate.png")
    print("  Saved: parallel_coordinate.png")
    
    # Slice plot
    fig_slice = plot_slice(study)
    fig_slice.write_image(f"./models/optuna_plots/{model_type}/slice_plot.png")
    print("  Saved: slice_plot.png")
    return None