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

* Evaluation uses a dataset-specific unit: TUEP patient or TUAB recording. The
  N epoch-level ``predict_proba`` outputs for that unit are averaged first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def find_optimal_threshold(subject_df: pd.DataFrame) -> float:
    """Find the macro-F1-optimal decision threshold on a subject-level DataFrame.

    Sweeps 181 evenly-spaced candidate thresholds in [0.05, 0.95] and returns
    the one that maximises macro-averaged F1.  Falls back to 0.5 if no
    threshold strictly improves over 0.5.

    Parameters
    ----------
    subject_df : pd.DataFrame
        As returned by :func:`subject_level_predict`; must have columns
        ``pred_proba`` and ``true_label``.

    Returns
    -------
    float
        Threshold in (0, 1).
    """
    y_true  = subject_df["true_label"].to_numpy()
    y_proba = subject_df["pred_proba"].to_numpy()
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
    return best_thresh


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    groups: list[str] | np.ndarray | None = None,
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

    # Class balancing: weight positive (control) examples inversely to their
    # frequency so the gradient is not dominated by the majority class.
    # Computed from epoch-level labels (proxy for subject-level ratio).
    counts = np.bincount(y_train.astype(int))
    if len(counts) >= 2 and counts[1] > 0:
        fixed_params["scale_pos_weight"] = float(counts[0]) / float(counts[1])

    # ── Phase 1: GridSearchCV ────────────────────────────────────────────────
    # Use a fixed large n_estimators; do NOT pass eval_set inside CV.
    phase1_params = {k: v for k, v in param_grid.items()
                     if k != "n_estimators"}
    phase1_params["n_estimators"] = [500]        # fix for search

    if groups is None:
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True,
            random_state=fixed_params.get("random_state", 42),
        )
    else:
        if len(groups) != len(y_train):
            raise ValueError("groups must contain one patient ID per training epoch")
        if len(set(groups)) < cv_folds:
            raise ValueError(
                f"XGBoost CV needs at least {cv_folds} training patients; "
                f"found {len(set(groups))}"
            )
        cv = StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True,
            random_state=fixed_params.get("random_state", 42),
        )

    # GPU context cannot be shared across sklearn parallel workers; run folds
    # sequentially when CUDA is the compute device.
    gs_n_jobs = 1 if fixed_params.get("device", "cpu") == "cuda" \
        else fixed_params.get("n_jobs", -1)

    base_model = xgb.XGBClassifier(**fixed_params)
    grid = GridSearchCV(
        base_model,
        phase1_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=gs_n_jobs,
        verbose=0,
        refit=False,           # we do the final fit ourselves
    )
    if groups is None:
        grid.fit(X_train, y_train)
    else:
        grid.fit(X_train, y_train, groups=np.asarray(groups))
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


def evaluation_level_predict(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    labels: np.ndarray,
    evaluation_ids: list[str],
    patient_ids: list[str],
    recording_ids: list[str],
    dataset_names: list[str],
) -> pd.DataFrame:
    """Average label-1 epoch probabilities within each evaluation unit."""
    n_rows = len(X)
    metadata = {
        "evaluation_ids": evaluation_ids,
        "patient_ids": patient_ids,
        "recording_ids": recording_ids,
        "dataset_names": dataset_names,
    }
    for name, values in metadata.items():
        if len(values) != n_rows:
            raise ValueError(f"{name} must contain one value per epoch")

    epoch_proba = model.predict_proba(X)[:, 1]
    epoch_df = pd.DataFrame({
        "dataset_name": dataset_names,
        "evaluation_id": evaluation_ids,
        "patient_id": patient_ids,
        "recording_id": recording_ids,
        "epoch_proba": epoch_proba,
        "true_label": labels,
    })
    for evaluation_id, group in epoch_df.groupby("evaluation_id"):
        if group["true_label"].nunique() != 1:
            raise ValueError(f"evaluation_id {evaluation_id!r} has multiple labels")
        if group["patient_id"].nunique() != 1:
            raise ValueError(f"evaluation_id {evaluation_id!r} has multiple patients")

    result = (
        epoch_df.groupby("evaluation_id", sort=True)
        .agg(
            dataset_name=("dataset_name", "first"),
            patient_id=("patient_id", "first"),
            recording_id=("recording_id", "first"),
            n_epochs=("epoch_proba", "size"),
            pred_proba=("epoch_proba", "mean"),
            true_label=("true_label", "first"),
        )
        .reset_index()
    )
    result.insert(1, "subject_id", result["evaluation_id"])
    # TUEP aggregation may span several recordings for one patient.
    result.loc[result["dataset_name"] == "tuep", "recording_id"] = ""
    return result


def evaluate_subject_level(
    subject_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute AUROC, macro-F1, and accuracy from subject-level predictions.

    Parameters
    ----------
    subject_df : pd.DataFrame
        As returned by :func:`subject_level_predict`.
    threshold : float
        Decision threshold for binary classification (default 0.5).  Pass the
        value returned by :func:`find_optimal_threshold` evaluated on the
        validation set to use an optimised cutpoint.

    Returns
    -------
    dict[str, float]
        Keys: ``"auroc"``, ``"f1"``, ``"accuracy"``, ``"threshold"``.
    """
    y_true  = subject_df["true_label"].to_numpy()
    y_proba = subject_df["pred_proba"].to_numpy()
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        "auroc":     float(roc_auc_score(y_true, y_proba)),
        "f1":        float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
