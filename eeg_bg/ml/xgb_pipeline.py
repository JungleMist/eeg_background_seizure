"""XGBoost training pipeline with GridSearchCV + early-stopping refit.

Design notes
------------
* **Two-phase training** avoids the incompatibility between
  ``GridSearchCV`` and XGBoost's ``eval_set`` early-stopping API:

  1. ``GridSearchCV`` (5-fold stratified, ``scoring="roc_auc"``) identifies
     the best hyperparameter combination.  ``n_estimators`` is fixed at a
     large value (500) during the search.
  2. The best hyperparameters are used to build a fresh estimator that is
     refitted on the full training set with ``eval_set=[(X_val, y_val)]``
     and early stopping, which determines the final number of trees.

* Evaluation is always at **subject level**: the N epoch-level ``predict_proba``
  outputs for each subject are averaged before computing metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier using grid search + early-stopping refit.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features ``(n_train, n_features)`` and labels.
    X_val, y_val : np.ndarray
        Validation features and labels — used *only* for early stopping in
        Phase 2 (never for hyperparameter selection).
    cfg : dict
        Loaded ``default.yaml``.  Reads ``cfg["ml"]["xgboost"]``.

    Returns
    -------
    xgb.XGBClassifier
        Fully fitted estimator (Phase 2 model, early-stopped on validation).
    """
    xgb_cfg          = cfg["ml"]["xgboost"]
    fixed_params     = dict(xgb_cfg["fixed"])
    param_grid       = dict(xgb_cfg["param_grid"])
    cv_folds         = int(xgb_cfg["cv_folds"])
    early_stop_rounds = int(xgb_cfg["early_stopping_rounds"])

    # Verbosity: suppress XGBoost internal output
    fixed_params.setdefault("verbosity", 0)

    # ── Phase 1: GridSearchCV ────────────────────────────────────────────────
    # Use a fixed large n_estimators; do NOT pass eval_set inside CV.
    phase1_params = {k: v for k, v in param_grid.items()
                     if k != "n_estimators"}
    phase1_params["n_estimators"] = [500]        # fix for search

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                          random_state=fixed_params.get("random_state", 42))

    base_model = xgb.XGBClassifier(**fixed_params)
    grid = GridSearchCV(
        base_model,
        phase1_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=fixed_params.get("n_jobs", -1),
        verbose=0,
        refit=False,           # we do the final fit ourselves
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_params.pop("n_estimators", None)  # will be determined by early stopping

    # ── Phase 2: Refit with early stopping on validation set ─────────────────
    final_model = xgb.XGBClassifier(
        **fixed_params,
        **best_params,
        early_stopping_rounds=early_stop_rounds,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return final_model


def subject_level_predict(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    subject_ids: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Average epoch-level probabilities to subject-level predictions.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Fitted classifier.
    X : np.ndarray, shape ``(n_epochs, n_features)``
    subject_ids : list[str]
        One subject ID per epoch row.
    labels : np.ndarray, shape ``(n_epochs,)``
        Ground-truth labels (0 = epilepsy, 1 = control).

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id``, ``pred_proba``, ``true_label``.
        One row per unique subject; ``pred_proba`` is the mean of all
        epoch-level ``predict_proba(X)[:, 1]`` for that subject.
    """
    epoch_proba = model.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "epoch_proba": epoch_proba,
        "true_label":  labels,
    })

    subject_df = (
        df.groupby("subject_id")
          .agg(pred_proba=("epoch_proba", "mean"),
               true_label=("true_label",  "first"))
          .reset_index()
    )
    return subject_df


def evaluate_subject_level(subject_df: pd.DataFrame) -> dict[str, float]:
    """Compute AUROC, macro-F1, and accuracy from subject-level predictions.

    Parameters
    ----------
    subject_df : pd.DataFrame
        As returned by :func:`subject_level_predict`.

    Returns
    -------
    dict[str, float]
        Keys: ``"auroc"``, ``"f1"``, ``"accuracy"``.
    """
    y_true  = subject_df["true_label"].to_numpy()
    y_proba = subject_df["pred_proba"].to_numpy()
    y_pred  = (y_proba >= 0.5).astype(int)

    return {
        "auroc":    float(roc_auc_score(y_true, y_proba)),
        "f1":       float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
