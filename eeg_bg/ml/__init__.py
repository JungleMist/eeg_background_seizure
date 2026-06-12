"""Machine-learning sub-package: XGBoost and CNN training pipelines, SHAP analysis.

Public API
----------
train_xgboost : callable
    GridSearchCV + early-stopping refit for XGBoost binary classification.
subject_level_predict : callable
    Average epoch-level probabilities to subject-level predictions.
evaluate_subject_level : callable
    Compute AUROC / F1 / Accuracy from subject-level predictions.
find_optimal_threshold : callable
    Find the F1-optimal decision threshold on a subject-level prediction DataFrame.
compute_shap_values : callable
    SHAP values via ``shap.TreeExplainer``.
aggregate_shap_by_band : callable
    Mean |SHAP| per EEG frequency-band feature group.
aggregate_shap_by_channel : callable
    Mean |SHAP| per EEG channel.
plot_shap_summary : callable
    Single-condition beeswarm summary figure.
plot_shap_comparison : callable
    2 x 3 cross-condition publication comparison figure.
EEGNet : class
    Compact EEGNet PyTorch module for binary EEG classification.
EEGEpochDataset : class
    PyTorch Dataset over the project's NPZ epoch cache.
cnn_predict_epochs : callable
    Run a fitted EEGNet over a DataLoader; return subject-level prediction DataFrame.
train_cnn : callable
    Full EEGNet training loop with early stopping; writes output files.
"""
from eeg_bg.ml.xgb_pipeline import (
    train_xgboost,
    subject_level_predict,
    evaluate_subject_level,
    find_optimal_threshold,
)
from eeg_bg.ml.shap_analysis import (
    compute_shap_values,
    aggregate_shap_by_band,
    aggregate_shap_by_channel,
    plot_shap_summary,
    plot_shap_comparison,
)
try:
    from eeg_bg.ml.cnn_model import EEGNet
    from eeg_bg.ml.cnn_dataset import EEGEpochDataset
    from eeg_bg.ml.cnn_pipeline import cnn_predict_epochs, train_cnn
except ImportError:
    pass  # torch not installed; CNN pipeline unavailable

__all__ = [
    # XGBoost pipeline
    "train_xgboost",
    "subject_level_predict",
    "evaluate_subject_level",
    "find_optimal_threshold",
    # SHAP analysis
    "compute_shap_values",
    "aggregate_shap_by_band",
    "aggregate_shap_by_channel",
    "plot_shap_summary",
    "plot_shap_comparison",
    # CNN pipeline
    "EEGNet",
    "EEGEpochDataset",
    "cnn_predict_epochs",
    "train_cnn",
]
