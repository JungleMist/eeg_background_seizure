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

# Also export per-epoch .edf files (up to export_edf_max_epochs >= 1 epochs/subject):
python scripts/05_run_visualization.py --export-edf --export-edf-max-epochs 3

Output
------
results/figures/{subject_id}/waveform_comparison.png
    1–3 side-by-side panels: Raw | Wiener specific | ICA cleaned
    All 19 channels stacked vertically; panels omitted if cache absent.

results/figures/{subject_id}/psd_comparison.png
    PSD overlay (raw / Wiener / ICA) for target channels (from config or --channels).

results/figures/{subject_id}/edf/epoch_{i}/{condition}.edf
    Only produced with --export-edf / visualization.export_edf: true. Up to
    export_edf_max_epochs (must be >= 1) epochs per subject, each reconstructed to a real
    .edf file per available condition (raw, wiener, wiener_phasegated,
    wiener_zerophase, ica), grouped in one folder per epoch so all
    preprocessing variants of the same original epoch can be compared side by
    side.
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


class VisualizationConfigError(ValueError):
    """Invalid visualization config or CLI option."""


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure to *path* at 150 dpi and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _validate_export_edf_max_epochs(value: int) -> int:
    """Return a positive EDF export cap or raise a clear config error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VisualizationConfigError(
            "export_edf_max_epochs must be a positive integer (>= 1)"
        )
    return value


def main(config_path: str, n_subjects: int | None, epoch_idx: int | None, channels: list[str] | None,
         export_edf: bool | None, export_edf_max_epochs: int | None) -> None:
    cfg         = load_config(config_path)
    channels    = channels    if channels    is not None else cfg["visualization"]["psd_target_channels"]
    n_subjects  = n_subjects  if n_subjects  is not None else cfg["visualization"]["n_subjects"]
    epoch_idx   = epoch_idx   if epoch_idx   is not None else cfg["visualization"]["epoch_idx"]
    export_edf  = export_edf  if export_edf  is not None else cfg["visualization"].get("export_edf", False)
    export_edf_max_epochs = (export_edf_max_epochs if export_edf_max_epochs is not None
                              else cfg["visualization"].get("export_edf_max_epochs", 3))
    if export_edf:
        export_edf_max_epochs = _validate_export_edf_max_epochs(export_edf_max_epochs)
    epoch_root     = Path(cfg["paths"]["cache_dir"]) / "epochs"
    wiener_root    = Path(cfg["paths"]["cache_dir"]) / "wiener_frequency"
    wiener_pg_root = Path(cfg["paths"]["cache_dir"]) / "wiener_phasegated"
    wiener_zp_root = Path(cfg["paths"]["cache_dir"]) / "wiener_zerophase"
    ica_root       = Path(cfg["paths"]["cache_dir"]) / "ica"
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
        edata        = np.load(npz_path, allow_pickle=True)
        legacy_id    = str(edata.get("subject_id", subj_dir.name))
        subj_id      = str(edata.get("evaluation_id", legacy_id))
        subj_fig_dir = fig_dir / subj_id
        subj_fig_dir.mkdir(parents=True, exist_ok=True)

        # ── Raw epoch ────────────────────────────────────────────────────────
        epochs   = edata["epochs"]            # (n_epochs, n_ch, n_times)
        ch_names = list(edata["ch_names"])
        ei       = min(epoch_idx, len(epochs) - 1)
        raw      = epochs[ei]                 # (n_ch, n_times)

        # ── Wiener-specific (optional) ────────────────────────────────────────
        wiener_path     = wiener_root / npz_path.relative_to(epoch_root)
        wiener_full     = None
        wiener_specific = None
        if wiener_path.exists():
            wdata           = np.load(wiener_path, allow_pickle=True)
            wiener_full     = wdata["specific"]       # (n_epochs, n_ch, n_times)
            wiener_specific = wiener_full[ei]         # (n_ch, n_times)

        # ── Wiener-phasegated (optional; EDF export only) ────────────────────
        wiener_pg_path = wiener_pg_root / npz_path.relative_to(epoch_root)
        wiener_pg_full = None
        if wiener_pg_path.exists():
            wiener_pg_full = np.load(wiener_pg_path, allow_pickle=True)["specific"]

        # ── Wiener-zerophase (optional) ───────────────────────────────────────
        wiener_zp_path = wiener_zp_root / npz_path.relative_to(epoch_root)
        wiener_zp_full = None
        if wiener_zp_path.exists():
            wiener_zp_full = np.load(wiener_zp_path, allow_pickle=True)["specific"]

        # ── ICA-cleaned (optional) ────────────────────────────────────────────
        ica_path     = ica_root / npz_path.relative_to(epoch_root)
        ica_full     = None
        ica_specific = None
        if ica_path.exists():
            idata        = np.load(ica_path, allow_pickle=True)
            ica_full     = idata["specific"]          # (n_epochs, n_ch, n_times)
            ica_specific = ica_full[ei]               # (n_ch, n_times)

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

        # ── EDF export (optional) ───────────────────────────────────────────
        if export_edf:
            from eeg_bg.io.edf_writer import export_epoch_edf

            n_export = min(export_edf_max_epochs, len(epochs))
            full_by_condition = {
                "raw": epochs, "wiener": wiener_full,
                "wiener_phasegated": wiener_pg_full,
                "wiener_zerophase": wiener_zp_full, "ica": ica_full,
            }
            for exp_ei in range(n_export):
                epoch_dir = subj_fig_dir / "edf" / f"epoch_{exp_ei}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                for cond_name, cond_full in full_by_condition.items():
                    if cond_full is None:
                        continue
                    export_epoch_edf(cond_full[exp_ei], ch_names, sfreq,
                                      epoch_dir / f"{cond_name}.edf")

    print(f"Done. Figures saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-channel waveform comparison figures"
    )
    parser.add_argument("--config",     default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Max number of subjects to process (0 = all); "
                             "default: visualization.n_subjects from config")
    parser.add_argument("--epoch-idx",  type=int, default=None,
                        help="Which epoch index to visualize per subject; "
                             "default: visualization.epoch_idx from config")
    parser.add_argument("--channels",   default=None,
                        help="Comma-separated channel names for PSD plot "
                             "(default: visualization.psd_target_channels from config)")
    parser.add_argument("--export-edf", dest="export_edf", action="store_true", default=None,
                        help="Also export each subject's epochs to .edf files per condition "
                             "(raw/wiener/wiener_phasegated/wiener_zerophase/ica); "
                             "default: visualization.export_edf from config (false)")
    parser.add_argument("--export-edf-max-epochs", type=int, default=None,
                        help="Positive max epochs per subject to export as .edf when --export-edf is set; "
                             "default: visualization.export_edf_max_epochs from config (3)")
    args = parser.parse_args()
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    try:
        main(args.config, args.n_subjects, args.epoch_idx, channels,
             args.export_edf, args.export_edf_max_epochs)
    except VisualizationConfigError as exc:
        parser.error(str(exc))
