"""SHAP-based explainability for XGBoost classifiers.

All plot functions save to disk and do **not** call ``plt.show()``.
The matplotlib ``Agg`` backend must be set before importing this module
(the calling script is responsible for that).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb

from eeg_bg.features.band_power import BANDS


# ── Internal helpers ─────────────────────────────────────────────────────────

def _mean_abs_shap_for(
    shap_values: np.ndarray,
    feature_names: list[str],
    pattern: str,
) -> float:
    """Mean |SHAP| across all features whose name contains *pattern*."""
    indices = [i for i, n in enumerate(feature_names) if pattern in n]
    if not indices:
        return 0.0
    return float(np.mean(np.abs(shap_values[:, indices])))


# ── Public API ────────────────────────────────────────────────────────────────

def compute_shap_values(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Compute SHAP values using ``TreeExplainer``.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Fitted XGBoost binary classifier.
    X : np.ndarray, shape ``(n_samples, n_features)``
    feature_names : list[str]
        Ordered feature names (e.g. :data:`eeg_bg.features.FEATURE_NAMES`).

    Returns
    -------
    np.ndarray, shape ``(n_samples, n_features)``
        SHAP values for the positive class (control / label=1).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.asarray(shap_values)


def aggregate_shap_by_band(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean |SHAP| aggregated by feature type.

    Groups features into: one group per EEG frequency band (``"_power"``
    suffix), one for Hjorth features, one for spectral entropy.

    Returns
    -------
    dict[str, float]
        Keys: ``"delta"``, ``"theta"``, ``"alpha"``, ``"beta"``,
        ``"gamma"``, ``"hjorth"``, ``"spectral_entropy"``.
    """
    result: dict[str, float] = {}
    for band in BANDS:
        result[band] = _mean_abs_shap_for(shap_values, feature_names,
                                           f"_{band}_power")
    result["hjorth"]           = _mean_abs_shap_for(shap_values, feature_names,
                                                     "_hjorth_")
    result["spectral_entropy"] = _mean_abs_shap_for(shap_values, feature_names,
                                                     "_spectral_entropy")
    result["asymmetry"]        = _mean_abs_shap_for(shap_values, feature_names,
                                                     "asym_")
    return result


def aggregate_shap_by_channel(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean |SHAP| aggregated across all 9 features of each channel.

    Returns
    -------
    dict[str, float]
        Keys are channel names (e.g. ``"FP1"``, ``"T3"``, …).
    """
    # Single pass: build {channel: [feature_indices]} index, then compute
    # mean |SHAP| per channel.  Asymmetry features ("asym_" prefix) are
    # excluded — they are reported under aggregate_shap_by_band("asymmetry").
    ch_indices: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        if name.startswith("asym_"):
            continue
        ch = name.split("_")[0]   # "FP1_delta_power" → "FP1"
        ch_indices.setdefault(ch, []).append(i)

    return {
        ch: float(np.mean(np.abs(shap_values[:, idxs])))
        for ch, idxs in ch_indices.items()
    }


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    title: str,
    output_path: Path,
    max_display: int = 20,
    dpi: int = 150,
) -> None:
    """Save a SHAP beeswarm summary plot for a single condition.

    Parameters
    ----------
    shap_values : np.ndarray, shape ``(n_samples, n_features)``
    X : np.ndarray, shape ``(n_samples, n_features)``
        Feature matrix (for colouring dots in the beeswarm).
    feature_names : list[str]
    title : str
        Figure title (condition label, e.g. ``"Wiener Specific (A)"``)
    output_path : Path
        Destination PNG file.
    max_display : int
        Number of top features to show.
    dpi : int
        Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_shap_comparison(
    results: dict[str, dict],
    output_path: Path,
    dpi: int = 200,
) -> None:
    """2 × 3 publication figure comparing SHAP importance across conditions.

    Parameters
    ----------
    results : dict[str, dict]
        Keys must include ``"raw"``, ``"ica"``, ``"wiener"``.  Each value is a
        dict with keys ``"shap_by_band"`` and ``"shap_by_channel"``.
    output_path : Path
        Destination PNG file.
    dpi : int
        Output resolution.

    Layout
    ------
    Row 1 (top):    Bar chart — mean |SHAP| per frequency-band feature group.
    Row 2 (bottom): Horizontal bar chart — mean |SHAP| per channel.

    Columns (left → right): Raw (C) | ICA (B) | Wiener (A).
    Same y-axis scale within each row for direct cross-condition comparison.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_order  = ["raw", "ica", "wiener"]
    condition_labels = ["Raw (C)", "ICA (B)", "Wiener (A)"]
    col_colors       = ["#888888", "#E07B54", "#4C8BBE"]

    band_keys = list(BANDS.keys()) + ["hjorth", "spectral_entropy", "asymmetry"]

    # Determine channel sort order from Wiener importance (fallback: raw).
    # Computed once here so all three columns share the same y-axis ordering.
    ref_ch = (results.get("wiener", results.get("raw", {}))
              .get("shap_by_channel", {}))
    ch_order = sorted(ref_ch, key=lambda c: ref_ch.get(c, 0), reverse=True)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        sharey="row",
    )

    for col_idx, (cond, label, color) in enumerate(
        zip(condition_order, condition_labels, col_colors)
    ):
        data = results.get(cond, {})

        # ── Row 0: band-level bar chart ───────────────────────────────────────
        ax0 = axes[0, col_idx]
        band_vals = [data.get("shap_by_band", {}).get(k, 0.0) for k in band_keys]
        ax0.bar(range(len(band_keys)), band_vals, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax0.set_xticks(range(len(band_keys)))
        ax0.set_xticklabels(band_keys, rotation=35, ha="right", fontsize=8)
        ax0.set_title(label, fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax0.set_ylabel("Mean |SHAP|", fontsize=9)
        ax0.grid(axis="y", alpha=0.3)
        ax0.tick_params(labelsize=8)

        # ── Row 1: channel-level horizontal bar chart ─────────────────────────
        ax1 = axes[1, col_idx]
        ch_data  = data.get("shap_by_channel", {})
        ch_vals = [ch_data.get(ch, 0.0) for ch in ch_order]
        y_pos   = range(len(ch_order))
        ax1.barh(list(y_pos), ch_vals, color=color, alpha=0.85,
                 edgecolor="white", linewidth=0.5)
        ax1.set_yticks(list(y_pos))
        ax1.set_yticklabels(ch_order, fontsize=7)
        ax1.invert_yaxis()
        if col_idx == 0:
            ax1.set_ylabel("Channel", fontsize=9)
        ax1.set_xlabel("Mean |SHAP|", fontsize=8)
        ax1.grid(axis="x", alpha=0.3)
        ax1.tick_params(labelsize=8)

    fig.suptitle(
        "SHAP Feature Importance: Raw (C) vs ICA (B) vs Wiener Specific (A)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
