import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import shap
from prefect import task
from omegaconf import DictConfig
import shap
from xgboost import XGBClassifier


@task
def plot_feature_importance(model: XGBClassifier) -> None:
    """Plot XGBoost feature importance using gain and cover metrics."""
    
    # Plot importance by gain
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, importance_type="gain")
    plt.title("Top 20 Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("./models/interpretability/feature_importance_gain.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot importance by cover
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, importance_type="cover")
    plt.title("Top 20 Feature Importance (Cover)")
    plt.tight_layout()
    plt.savefig("./models/interpretability/feature_importance_cover.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature importance plots saved!")
    return None


@task
def shap_summary_plot(model: XGBClassifier, data: pd.DataFrame, sample_size: int = 1000) -> None:
    """Generate SHAP summary plot showing feature importance and impact direction."""
    
    # Sample data for faster computation
    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for SHAP computation (out of {len(data)} total)")
    else:
        data_sample = data
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_sample)
    
    # Summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, data_sample, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("./models/interpretability/shap_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, data_sample, show=False, max_display=20)
    plt.title("SHAP Summary Plot (Feature Impact)")
    plt.tight_layout()
    plt.savefig("./models/interpretability/shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SHAP summary plots saved!")
    return None


@task
def shap_waterfall_plot(model: XGBClassifier, data: pd.DataFrame, sample_idx: int = 0,sample_size: int = 1000) -> None:
    """Generate SHAP waterfall plot for a single prediction."""
    
    # Sample data for faster computation
    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for SHAP computation (out of {len(data)} total)")
    else:
        data_sample = data
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data_sample)
    
    # Waterfall plot for a specific sample
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=15)
    plt.title(f"SHAP Waterfall Plot (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(f"./models/interpretability/shap_waterfall_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP waterfall plot for sample {sample_idx} saved!")
    return None


@task
def shap_dependence_plots(model: XGBClassifier, data: pd.DataFrame, top_n_features: int = 5, sample_size: int = 1000) -> None:
    """Generate SHAP dependence plots for top N features."""
    
    # Sample data for faster computation
    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for SHAP dependence plots (out of {len(data)} total)")
    else:
        data_sample = data
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_sample)
    
    # Get feature importance to find top features
    feature_importance = pd.DataFrame({
        'feature': data_sample.columns,
        'importance': abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(top_n_features)['feature'].tolist()
    
    # Create dependence plots for top features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, data_sample, show=False)
        plt.title(f"SHAP Dependence Plot: {feature}")
        plt.tight_layout()
        # Clean feature name for filename
        safe_feature_name = feature.replace('/', '_').replace(' ', '_')
        plt.savefig(f"./models/interpretability/shap_dependence_{safe_feature_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"SHAP dependence plots for top {top_n_features} features saved!")
    return None

@task
def roc_plots(test_features, test_target):
       # Generate predictions and probabilities
    y_pred_proba = model.predict_proba(test_features)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(test_target, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    pl.figure(figsize=(10, 8))
    pl.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    pl.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver Operating Characteristic (ROC) Curve')
    pl.legend(loc="lower right")
    pl.grid(alpha=0.3)
    pl.tight_layout()
    pl.savefig("./models/interpretability/roc_curve.png", dpi=300, bbox_inches='tight')
    pl.close()
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("ROC curve saved to ./models/interpretability/roc_curve.png")
    return None
