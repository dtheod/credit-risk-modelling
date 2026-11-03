"""
Out-of-time test-set evaluation for the PD and LGD models.

Metrics are logged to MLflow and written to JSON so the numbers quoted in the README can be
traced back to a pipeline run.
"""

import json
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from prefect import task
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def _save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


@task
def evaluate_pd_model(y_true: pd.Series, y_proba: np.ndarray, country: pd.Series,
                      metrics_path: str, plot_path: str) -> dict:
    """
    Scores the PD model on the out-of-time test set: AUC, Gini, KS, Brier score, and AUC
    within each country (to show how much of the ranking power is the country effect).
    """
    y_true = np.asarray(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    auc_by_country = {}
    for c in sorted(pd.unique(country)):
        mask = np.asarray(country == c)
        if len(np.unique(y_true[mask])) == 2:
            auc_by_country[c] = round(float(roc_auc_score(y_true[mask], y_proba[mask])), 4)

    metrics = {
        "test_roc_auc": round(float(auc), 4),
        "test_gini": round(float(2 * auc - 1), 4),
        "test_ks": round(float(np.max(tpr - fpr)), 4),
        "test_brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "test_default_rate": round(float(y_true.mean()), 4),
        "test_mean_predicted_pd": round(float(np.mean(y_proba)), 4),
        "test_n": int(len(y_true)),
    }
    mlflow.log_metrics(metrics)
    mlflow.log_metrics({f"test_roc_auc_{c}": v for c, v in auc_by_country.items()})
    metrics["test_roc_auc_by_country"] = auc_by_country
    _save_metrics(metrics, metrics_path)

    # ROC + calibration plot
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (out-of-time test set)")
    ax1.legend(loc="lower right")
    ax2.plot(prob_pred, prob_true, marker="o", lw=2, label="Model (deciles)")
    lim = max(prob_pred.max(), prob_true.max()) * 1.05
    ax2.plot([0, lim], [0, lim], linestyle="--", color="grey", label="Perfect calibration")
    ax2.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title="Calibration (out-of-time test set)")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Test ROC AUC: {auc:.4f} | by country: {auc_by_country}")
    return metrics


@task
def evaluate_lgd_model(y_true: pd.Series, y_pred: np.ndarray, train_mean: float, metrics_path: str) -> dict:
    """
    Scores the LGD model on the out-of-time test set. `baseline_mae` is the MAE of always
    predicting the training-set mean LGD, as a reference point.
    """
    metrics = {
        "test_mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "test_r2": round(float(r2_score(y_true, y_pred)), 4),
        "test_spearman": round(float(spearmanr(y_true, y_pred).statistic), 4),
        "test_pearson": round(float(pearsonr(y_true, y_pred).statistic), 4),
        "test_baseline_mae": round(float(mean_absolute_error(y_true, np.full(len(y_true), train_mean))), 4),
        "test_mean_lgd": round(float(np.mean(y_true)), 4),
        "test_n": int(len(y_true)),
    }
    mlflow.log_metrics(metrics)
    _save_metrics(metrics, metrics_path)
    print(f"LGD test metrics: {metrics}")
    return metrics
