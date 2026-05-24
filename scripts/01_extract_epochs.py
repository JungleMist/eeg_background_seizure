"""Extract background epochs from all EDF files and cache to disk."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.io.annotation import extract_bckg_intervals
from eeg_bg.io.cache import make_cache_key
from eeg_bg.io.dataset import assign_splits, build_subject_index
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.preprocessing.epoch import slice_epochs
from eeg_bg.preprocessing.reference import filter_by_reference


def _init_worker():
    import mne
    mne.set_log_level("ERROR")


def _process_one_row(args):
    row_dict, cfg, cache_root_str, force = args
    edf_path = Path(row_dict["edf_path"])
    csv_bi_path = edf_path.with_suffix(".csv_bi")

    cache_key = make_cache_key(edf_path, 0.0, cfg)
    label_prefix = "00" if row_dict["label"] == 0 else "01"
    prefixed_sid = f"{label_prefix}_{row_dict['subject_id']}"
    cache_path = Path(cache_root_str) / prefixed_sid / f"{cache_key}.npz"

    if cache_path.exists() and not force:
        return (edf_path.name, "cached", None)

    try:
        data, ch_names, sfreq = load_edf(edf_path, cfg)
    except Exception as e:
        return (edf_path.name, "skip", str(e))

    duration = data.shape[1] / sfreq
    intervals = (
        extract_bckg_intervals(csv_bi_path, cfg, recording_duration=duration)
        if csv_bi_path.exists()
        else [(0.0, duration)]
    )

    epochs = slice_epochs(
        data, sfreq, intervals,
        cfg["preprocessing"]["epoch_length_sec"],
        cfg["preprocessing"]["artifact_threshold_uv"],
    )

    if epochs.shape[0] == 0:
        return (edf_path.name, "skip", "no valid epochs")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        epochs=epochs,
        ch_names=np.array(ch_names),
        label=np.array(row_dict["label"]),
        subject_id=np.array(prefixed_sid),
        split=np.array(row_dict["split"]),
    )
    return (edf_path.name, "done", None)


def main(config_path: str, force: bool = False, max_workers: int | None = None) -> None:
    cfg = load_config(config_path)
    cache_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    cache_root.mkdir(parents=True, exist_ok=True)

    index = build_subject_index(cfg)
    index = filter_by_reference(index, cfg["dataset"]["reference_scheme"])
    index = assign_splits(index, cfg)
    index.to_csv(cache_root / "index.csv", index=False)
    print(f"Found {len(index)} EDF files across {index['subject_id'].nunique()} subjects.")

    args_list = [
        (row.to_dict(), cfg, str(cache_root), force)
        for _, row in index.iterrows()
    ]

    n_workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_one_row, args): args for args in args_list}
        with tqdm(total=len(futures), desc="Extracting epochs") as pbar:
            for future in as_completed(futures):
                fname, status, msg = future.result()
                if status == "skip":
                    tqdm.write(f"  SKIP {fname}: {msg}")
                pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract background EEG epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()
    main(args.config, args.force, args.workers)
