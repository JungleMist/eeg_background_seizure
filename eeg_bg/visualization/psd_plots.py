from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def plot_psd_comparison(
    raw: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    channels: list[str],
    wiener_specific: np.ndarray | None = None,
    ica_specific: np.ndarray | None = None,
    nperseg: int = 250,
    freq_band: tuple[float, float] = (0.5, 40.0),
    title: str = "",
) -> plt.Figure:
    """PSD comparison: rows = target channels, columns = Raw | Wiener Specific | ICA Cleaned.

    Parameters
    ----------
    raw:             (n_ch, n_times) raw epoch in µV
    ch_names:        channel name list matching raw's first axis
    sfreq:           sampling frequency in Hz
    channels:        channel names to plot (one row per channel)
    wiener_specific: (n_ch, n_times) Wiener-cleaned epoch, or None to show placeholder
    ica_specific:    (n_ch, n_times) ICA-cleaned epoch, or None to show placeholder
    nperseg:         Welch window length (boxcar, consistent with decomposition)
    freq_band:       (low, high) Hz — x-axis limits
    title:           figure suptitle
    """
    col_labels = ["Raw", "Wiener Specific", "ICA Cleaned"]
    col_data = [raw, wiener_specific, ica_specific]
    col_colors = ["gray", "steelblue", "coral"]
    n_rows = len(channels)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 2.5 * n_rows),
        sharey="row",
        sharex=True,
        squeeze=False,
    )

    for row, ch in enumerate(channels):
        try:
            idx = ch_names.index(ch)
        except ValueError:
            for col in range(n_cols):
                axes[row, col].set_visible(False)
            ax0 = axes[row, 0]
            ax0.set_visible(True)
            ax0.text(0.5, 0.5, f"{ch} (not found)",
                     ha="center", va="center", fontsize=8,
                     transform=ax0.transAxes, color="gray")
            continue

        for col, (data, color, col_label) in enumerate(
            zip(col_data, col_colors, col_labels)
        ):
            ax = axes[row, col]

            if data is None:
                ax.text(0.5, 0.5, "not available",
                        ha="center", va="center", fontsize=8,
                        transform=ax.transAxes, color="gray")
                ax.set_xlim(freq_band)
            else:
                f, p = welch(data[idx], fs=sfreq, nperseg=nperseg, window="boxcar")
                mask = (f >= freq_band[0]) & (f <= freq_band[1])
                ax.plot(f[mask], 10 * np.log10(p[mask] + 1e-30),
                        color=color, lw=0.9)
                ax.set_xlim(freq_band)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_label, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{ch}\nPower (dB rel. µV²/Hz)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
