"""Multi-channel waveform comparison plots."""
import numpy as np
import matplotlib.pyplot as plt


def plot_multichannel_comparison(
    raw: np.ndarray,
    wiener_specific: np.ndarray | None,
    ica_specific: np.ndarray | None,
    ch_names: list[str],
    sfreq: float = 125.0,
    offset_uv: float = 150.0,
    title: str = "",
) -> plt.Figure:
    """Stacked multi-channel EEG waveform comparison.

    Produces 1–3 side-by-side panels depending on which arrays are provided.
    Raw is always shown. Wiener-specific and ICA-cleaned panels are added when
    their arrays are not None. Channels are stacked vertically with a fixed µV
    offset so traces are visually separable.

    Parameters
    ----------
    raw : ndarray, shape (n_ch, n_times)
        Original EEG signal in µV.
    wiener_specific : ndarray (n_ch, n_times) or None
        Source-specific component after Wiener decomposition (µV).
    ica_specific : ndarray (n_ch, n_times) or None
        ICA-cleaned signal (µV).
    ch_names : list of str
        Channel labels, length n_ch. Listed top-to-bottom on y-axis.
    sfreq : float
        Sampling frequency in Hz (default 125).
    offset_uv : float
        Vertical separation between adjacent channel traces in µV (default 150).
    title : str
        Figure suptitle (e.g. "subject_id  epoch 0").

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels: list[tuple[str, np.ndarray]] = [("Raw", raw)]
    if wiener_specific is not None:
        panels.append(("Wiener specific", wiener_specific))
    if ica_specific is not None:
        panels.append(("ICA cleaned", ica_specific))

    n_panels = len(panels)
    n_ch, n_times = raw.shape
    t = np.arange(n_times) / sfreq
    # Channel 0 at top → highest offset; channel n_ch-1 at bottom → offset 0
    offsets = np.arange(n_ch - 1, -1, -1) * offset_uv

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, max(6, 0.45 * n_ch)),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (panel_title, data) in zip(axes, panels):
        for i, off in enumerate(offsets):
            ax.plot(t, data[i] + off, lw=0.6, color="steelblue")
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(ch_names, fontsize=7)
        ax.grid(axis="x", lw=0.3, alpha=0.5)

    axes[0].set_ylabel("Channel")
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig
