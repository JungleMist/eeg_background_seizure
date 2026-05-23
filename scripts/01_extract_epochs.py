"""Extract background epochs from all EDF files and cache to disk."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import build_subject_index, assign_splits
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.io.annotation import extract_bckg_intervals
from eeg_bg.io.cache import make_cache_key
from eeg_bg.preprocessing.epoch import bandpass_filter, slice_epochs
from eeg_bg.preprocessing.reference import filter_by_reference


def main(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    cache_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    cache_root.mkdir(parents=True, exist_ok=True)

    index = build_subject_index(cfg)
    index = filter_by_reference(index, cfg["dataset"]["reference_scheme"])
    index = assign_splits(index, cfg)
    index.to_csv(cache_root / "index.csv", index=False)
    print(f"Found {len(index)} EDF files across {index['subject_id'].nunique()} subjects.")

    for _, row in tqdm(index.iterrows(), total=len(index), desc="Extracting epochs"):
        edf_path = Path(row["edf_path"])
        csv_bi_path = edf_path.with_suffix(".csv_bi")

        cache_key = make_cache_key(edf_path, 0.0, cfg)
        label_prefix = "00" if row["label"] == 0 else "01"
        prefixed_sid = f"{label_prefix}_{row['subject_id']}"
        cache_path = cache_root / prefixed_sid / f"{cache_key}.npz"

        if cache_path.exists() and not force:
            continue

        try:
            data, ch_names, sfreq = load_edf(edf_path, cfg)
        except Exception as e:
            print(f"  SKIP {edf_path.name}: {e}")
            continue

        if csv_bi_path.exists():
            intervals = extract_bckg_intervals(csv_bi_path, cfg)
        else:
            duration = data.shape[1] / sfreq
            intervals = [(0.0, duration)]

        epochs = slice_epochs(
            data, sfreq, intervals,
            cfg["preprocessing"]["epoch_length_sec"],
            cfg["preprocessing"]["artifact_threshold_uv"],
        )

        if epochs.shape[0] == 0:
            print(f"  SKIP {edf_path.name}: no valid epochs")
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            epochs=epochs,
            ch_names=np.array(ch_names),
            label=np.array(row["label"]),
            subject_id=np.array(prefixed_sid),
            split=np.array(row["split"]),
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract background EEG epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    args = parser.parse_args()
    main(args.config, args.force)
