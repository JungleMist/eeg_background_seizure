"""Generate per-subject multi-channel waveform and PSD comparison figures.

For each subject, loads one representative epoch from the epoch cache and the
corresponding Wiener / ICA caches (when available), then saves figures to
results/figures/{subject_id}/.

Usage
-----
# All subjects, epoch 0, default PSD channels from config:
python scripts/05_run_visualization.py

# Limit to first 5 subjects, epoch 2, custom PSD channels:
python scripts/05_run_visualization.py --n-subjects 5 --epoch-idx 2 --channels "FP1,FP2,T3"

Output
------
results/figures/{subject_id}/waveform_comparison.png
    1–3 side-by-side panels: Raw | Wiener specific | ICA cleaned
    All 19 channels stacked vertically; panels omitted if cache absent.

results/figures/{subject_id}/psd_comparison.png
    PSD overlay (raw / Wiener / ICA) for target channels (from config or --channels).
"""
import argparse
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.visualization.waveform_plots import plot_multichannel_comparison
from eeg_bg.visualization.psd_plots import plot_psd_comparison


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure to *path* at 150 dpi and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str, n_subjects: int, epoch_idx: int, channels: list[str] | None) -> None:
    cfg         = load_config(config_path)
    channels    = channels or cfg["visualization"]["psd_target_channels"]
    epoch_root  = Path(cfg["paths"]["cache_dir"]) / "epochs"
    wiener_root = Path(cfg["paths"]["cache_dir"]) / "wiener_frequency"
    ica_root    = Path(cfg["paths"]["cache_dir"]) / "ica"
    fig_dir     = Path(cfg["paths"]["results_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sfreq       = float(cfg["preprocessing"]["target_sfreq"])

    subject_dirs = sorted(d for d in epoch_root.iterdir() if d.is_dir())
    if n_subjects > 0:
        subject_dirs = subject_dirs[:n_subjects]

    if not subject_dirs:
        print("No subject directories found in epoch cache. Run 01_extract_epochs.py first.")
        return

    for subj_dir in tqdm(subject_dirs, desc="Subjects"):
        npz_files = sorted(subj_dir.glob("*.npz"))
        if not npz_files:
            continue

        npz_path     = npz_files[0]          # first session .npz
        subj_id      = subj_dir.name
        subj_fig_dir = fig_dir / subj_id
        subj_fig_dir.mkdir(parents=True, exist_ok=True)

        # ── Raw epoch ────────────────────────────────────────────────────────
        edata    = np.load(npz_path, allow_pickle=True)
        epochs   = edata["epochs"]            # (n_epochs, n_ch, n_times)
        ch_names = list(edata["ch_names"])
        ei       = min(epoch_idx, len(epochs) - 1)
        raw      = epochs[ei]                 # (n_ch, n_times)

        # ── Wiener-specific (optional) ────────────────────────────────────────
        wiener_path     = wiener_root / npz_path.relative_to(epoch_root)
        wiener_specific = None
        if wiener_path.exists():
            wdata           = np.load(wiener_path, allow_pickle=True)
            wiener_specific = wdata["specific"][ei]   # (n_ch, n_times)

        # ── ICA-cleaned (optional) ────────────────────────────────────────────
        ica_path     = ica_root / npz_path.relative_to(epoch_root)
        ica_specific = None
        if ica_path.exists():
            idata        = np.load(ica_path, allow_pickle=True)
            ica_specific = idata["specific"][ei]      # (n_ch, n_times)

        fig = plot_multichannel_comparison(
            raw, wiener_specific, ica_specific,
            ch_names,
            sfreq=sfreq,
            title=f"{subj_id}  epoch {ei}",
        )
        _save(fig, subj_fig_dir / "waveform_comparison.png")

        fig_psd = plot_psd_comparison(
            raw, ch_names, sfreq,
            channels=channels,
            wiener_specific=wiener_specific,
            ica_specific=ica_specific,
            nperseg=cfg["wiener"]["nperseg"],
            freq_band=tuple(cfg["wiener"]["freq_band"]),
            title=f"{subj_id}  epoch {ei}",
        )
        _save(fig_psd, subj_fig_dir / "psd_comparison.png")

    print(f"Done. Figures saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-channel waveform comparison figures"
    )
    parser.add_argument("--config",     default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--n-subjects", type=int, default=0,
                        help="Max number of subjects to process (0 = all)")
    parser.add_argument("--epoch-idx",  type=int, default=0,
                        help="Which epoch index to visualize per subject")
    parser.add_argument("--channels",   default=None,
                        help="Comma-separated channel names for PSD plot "
                             "(default: visualization.psd_target_channels from config)")
    args = parser.parse_args()
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    main(args.config, args.n_subjects, args.epoch_idx, channels)
