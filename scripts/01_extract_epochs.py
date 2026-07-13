"""Extract dataset-specific background epochs from EDF files."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.io.cache import build_cache_metadata, make_cache_key
from eeg_bg.io.dataset import (
    active_dataset_name,
    assign_dataset_splits,
    build_recording_index,
    get_recording_intervals,
)
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.preprocessing.epoch import slice_epochs


def _init_worker():
    import mne
    mne.set_log_level("ERROR")


def _process_one_row(args):
    row, cfg, cache_root_str, force = args
    edf_path = Path(row["edf_path"])
    cache_key = make_cache_key(edf_path, 0.0, cfg)
    cache_path = Path(cache_root_str) / row["evaluation_id"] / f"{cache_key}.npz"

    if cache_path.exists() and not force:
        return edf_path.name, "cached", None, 0

    try:
        data, ch_names, sfreq = load_edf(edf_path, cfg)
        duration = data.shape[1] / sfreq
        intervals = get_recording_intervals(row, cfg, duration)
        epochs = slice_epochs(
            data,
            sfreq,
            intervals,
            cfg["preprocessing"]["epoch_length_sec"],
            cfg["preprocessing"]["artifact_threshold_uv"],
        )
    except Exception as exc:
        return edf_path.name, "skip", str(exc), 0

    if epochs.shape[0] == 0:
        return edf_path.name, "skip", "no valid epochs", 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        epochs=epochs,
        ch_names=np.asarray(ch_names),
        **build_cache_metadata(row, len(epochs)),
    )
    return edf_path.name, "done", None, len(epochs)


def _print_index_warnings(index, dataset_name: str) -> None:
    if index.empty:
        print("[WARNING] No EDF recordings found for the active dataset.")
        return
    if dataset_name == "tuab":
        for partition in ("train", "test"):
            if not (index["split"] == partition).any():
                print(f"[WARNING] TUAB index contains no {partition} recordings.")
        for class_name in ("abnormal", "normal"):
            if not (index["class_name"] == class_name).any():
                print(f"[WARNING] TUAB index contains no {class_name} recordings.")


def main(config_path: str, force: bool = False, max_workers: int | None = None) -> None:
    cfg = load_config(config_path)
    dataset_name = active_dataset_name(cfg)
    cache_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    cache_root.mkdir(parents=True, exist_ok=True)

    index = assign_dataset_splits(build_recording_index(cfg), cfg)
    index.to_csv(cache_root / "index.csv", index=False)
    _print_index_warnings(index, dataset_name)
    print(
        f"Found {len(index)} {dataset_name.upper()} EDF files across "
        f"{index['patient_id'].nunique() if not index.empty else 0} patients and "
        f"{index['evaluation_id'].nunique() if not index.empty else 0} evaluation units."
    )
    if index.empty:
        return

    args_list = [
        (row.to_dict(), cfg, str(cache_root), force)
        for _, row in index.iterrows()
    ]
    status_counts = {"done": 0, "cached": 0, "skip": 0}
    total_epochs = 0
    n_workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_one_row, args): args for args in args_list}
        with tqdm(total=len(futures), desc="Extracting epochs") as pbar:
            for future in as_completed(futures):
                fname, status, msg, n_epochs = future.result()
                status_counts[status] += 1
                total_epochs += n_epochs
                if status == "skip":
                    tqdm.write(f"  SKIP {fname}: {msg}")
                pbar.update(1)

    print(
        "Done. "
        f"processed={status_counts['done']}, cached={status_counts['cached']}, "
        f"skipped={status_counts['skip']}, new_epochs={total_epochs}."
    )


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
