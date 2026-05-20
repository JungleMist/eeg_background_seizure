from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from eeg_bg.decomposition.wiener import WienerResult


def plot_wiener_filter_response(
    result: WienerResult,
    pair_key: str,
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    if pair_key not in result.filters:
        raise KeyError(f"Pair '{pair_key}' not found. Available: {list(result.filters.keys())}")
    ch_name = list(result.filters[pair_key].keys())[0]
    h = result.filters[pair_key][ch_name][0]  # (n_freqs,)
    freqs = result.freqs

    if ax is None:
        fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    else:
        fig = ax[0].figure
        ax_amp, ax_phase = ax

    ax_amp.plot(freqs, np.abs(h))
    ax_amp.set_ylabel("|h(f)|")
    ax_amp.set_title(f"Wiener filter response: {pair_key}")
    ax_amp.grid(True, alpha=0.3)

    ax_phase.plot(freqs, np.angle(h))
    ax_phase.set_ylabel("angle h(f) [rad]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_xlim(0, 45)
    ax_phase.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_pairs_response(
    result: WienerResult,
    save_path=None,
) -> plt.Figure:
    pairs = list(result.filters.keys())
    n = len(pairs)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No filter results", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 5), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, pair_key in enumerate(pairs):
        plot_wiener_filter_response(result, pair_key, ax=(axes[0, col], axes[1, col]))
        axes[0, col].set_title(pair_key, fontsize=9)

    fig.suptitle("Wiener filter responses -- all bilateral pairs", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
