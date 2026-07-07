# Experiment: 2026-07-07_104536_default_wiener-zerophase

**Date:** 2026-07-07 10:45:36  
**Config:** configs/default.yaml  
**Conditions found:** raw, ica, wiener, wiener_zerophase

## Configuration Snapshot

| Parameter | Value |
|---|---|
| `target_sfreq` | 125 |
| `bandpass` | [0.5, 40.0] |
| `epoch_length_sec` | 20.0 |
| `artifact_threshold_uv` | 200.0 |
| `seizure_buffer_sec` | 30.0 |
| `split.train` | 0.7 |
| `split.val` | 0.1 |
| `split.test` | 0.2 |
| `split.random_seed` | 42 |
| `wiener.mode` | frequency |
| `wiener.nperseg` | 500 |
| `wiener.freq_resolution_hz` | 0.5 |
| `wiener.coherence_threshold` | 0.15 |
| `ica.n_components` | 19 |
| `ica.artifact_corr_threshold` | 0.8 |
| `ml.cv_folds` | 5 |
| `ml.early_stopping_rounds` | 30 |
| `ml.param_grid` | {"max_depth": [3, 4, 5, 6], "learning_rate": [0.01, 0.05, 0.1, 0.3], "n_estimators": [100, 200, 500], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "reg_alpha": [0.0, 0.1], "reg_lambda": [1.0, 5.0]} |

## XGBoost Results Summary

| Condition | Val AUROC | Val F1 | Val Acc | Test AUROC | Test F1 | Test Acc |
|---|---|---|---|---|---|---|
| raw | 0.542 | 0.582 | 0.588 | 0.738 | 0.725 | 0.730 |
| ica | 0.486 | 0.564 | 0.588 | 0.721 | 0.563 | 0.595 |
| wiener | 0.556 | 0.614 | 0.647 | 0.744 | 0.598 | 0.622 |
| wiener_zerophase | 0.583 | 0.702 | 0.706 | 0.653 | 0.619 | 0.622 |

## Dataset Statistics

| Split | Subjects | Epochs | Epilepsy Subj | Control Subj | Epilepsy Ep | Control Ep |
|---|---|---|---|---|---|---|
| train | 124 | 149224 | 67 | 57 | 119443 | 29781 |
| val | 17 | 5526 | 9 | 8 | 3811 | 1715 |
| test | 37 | 26346 | 20 | 17 | 21952 | 4394 |

## SHAP Comparison

![SHAP Comparison](shap_comparison.png)

## Per-Condition SHAP Summaries

### Raw

![Raw SHAP](raw/shap_summary.png)

### Ica

![Ica SHAP](ica/shap_summary.png)

### Wiener

![Wiener SHAP](wiener/shap_summary.png)

### Wiener_zerophase

![Wiener_zerophase SHAP](wiener_zerophase/shap_summary.png)
