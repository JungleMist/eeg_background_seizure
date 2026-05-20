from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eeg_bg.decomposition.wiener import WienerResult


def plot_coherence_matrix(
    coh_pre: np.ndarray,
    coh_post: np.ndarray,
    ch_names: list[str],
    title: str = "",
) -> plt.Figure:
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, label in [(ax_pre, coh_pre, "Before decomposition"),
                            (ax_post, coh_post, "After decomposition")]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="hot_r", aspect="auto")
        ax.set_xticks(range(len(ch_names)))
        ax.set_yticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
        ax.set_yticklabels(ch_names, fontsize=8)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_coherence_reduction(
    v1_df: pd.DataFrame,
    group_by: str = "pair",
) -> plt.Figure:
    df = v1_df.copy()
    if group_by == "pair":
        df["pair"] = df["ch_i"] + "-" + df["ch_j"]
        x_col = "pair"
    else:
        x_col = "subject_id"

    groups = sorted(df[x_col].unique())
    data = [df[df[x_col] == g]["reduction"].values for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6), 5))
    ax.boxplot(data, labels=groups, patch_artist=True)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Coherence reduction (pre - post)")
    ax.set_title("V1: Coherence reduction by " + x_col)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_signal_decomposition(
    result: WienerResult,
    ch_name: str,
    sfreq: float = 125.0,
    epoch_idx: int = 0,
    time_window: tuple[float, float] = (0.0, 4.0),
) -> plt.Figure:
    ch_idx = result.ch_names.index(ch_name)
    t = np.arange(result.raw.shape[1]) / sfreq
    mask = (t >= time_window[0]) & (t <= time_window[1])

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, sig, label in [
        (axes[0], result.raw[ch_idx], "Raw x(t)"),
        (axes[1], result.coherent[ch_idx], "Coherent component"),
        (axes[2], result.specific[ch_idx], "Specific component"),
    ]:
        ax.plot(t[mask], sig[mask])
        ax.set_ylabel(label + " [uV]")
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Time [s]")
    axes[0].set_title(f"Signal decomposition -- {ch_name}, epoch {epoch_idx}")
    fig.tight_layout()
    return fig
