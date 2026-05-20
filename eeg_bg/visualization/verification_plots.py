from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_v2_transitivity(v2_df: pd.DataFrame) -> plt.Figure:
    fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))
    ax_amp.hist(v2_df["eps_amp"], bins=30, edgecolor="k", alpha=0.7)
    ax_amp.axvline(0.1, color="r", linestyle="--", label="threshold 0.1")
    ax_amp.set_xlabel("eps_amplitude")
    ax_amp.set_title("V2: Transitivity amplitude error")
    ax_amp.legend()

    ax_phase.hist(v2_df["eps_phase"], bins=30, edgecolor="k", alpha=0.7)
    ax_phase.axvline(0.392, color="r", linestyle="--", label="threshold pi/8")
    ax_phase.set_xlabel("eps_phase [rad]")
    ax_phase.set_title("V2: Transitivity phase error")
    ax_phase.legend()

    fig.tight_layout()
    return fig


def plot_v3_frequency_variation(v3_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    pairs = v3_df["pair"].unique()
    for i, pair in enumerate(pairs):
        grp = v3_df[v3_df["pair"] == pair]
        ax.bar([i], [grp["freq_variation"].mean()],
               yerr=[grp["freq_variation"].std()], capsize=4, alpha=0.7,
               label=pair)
    ax.axhline(0.20, color="r", linestyle="--", label="threshold 20%")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel("Relative frequency variation of |h(f)|")
    ax.set_title("V3: Frequency-dependence of Wiener filter")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ica_vs_wiener_coherence(
    raw_pre: np.ndarray,
    ica_post: np.ndarray,
    wiener_post: np.ndarray,
    ch_names: list[str],
) -> plt.Figure:
    from eeg_bg.verification.coherence import compute_pairwise_coherence
    sfreq, nperseg, freq_band = 125.0, 500, (0.5, 40.0)

    coh_raw = compute_pairwise_coherence(raw_pre, sfreq, nperseg, freq_band)
    coh_ica = compute_pairwise_coherence(ica_post, sfreq, nperseg, freq_band)
    coh_wiener = compute_pairwise_coherence(wiener_post, sfreq, nperseg, freq_band)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title in [
        (axes[0], coh_raw, "Raw"),
        (axes[1], coh_ica, "ICA"),
        (axes[2], coh_wiener, "Wiener"),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="hot_r", aspect="auto")
        ax.set_xticks(range(len(ch_names)))
        ax.set_yticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_yticklabels(ch_names, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Pairwise coherence: Raw vs ICA vs Wiener", fontsize=12)
    fig.tight_layout()
    return fig
