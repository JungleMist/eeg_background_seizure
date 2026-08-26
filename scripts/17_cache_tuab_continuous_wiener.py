"""Cache continuous TUAB ECMAD specific and coherent component sequences."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.io.cache import make_wiener_cache_fingerprint
from eeg_bg.io.dataset import (
    active_dataset_config,
    active_dataset_name,
    assign_dataset_splits,
    build_recording_index,
)
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.preprocessing.continuous import wiener_continuous_raw


CACHE_SCHEMA_VERSION = 1
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
COMPONENTS = ("specific", "coherent")


class CacheFingerprintMismatch(ValueError):
    """Raised when a complete cache belongs to a different source/config."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(_jsonable(payload), sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_paths(output_root: Path, evaluation_id: str) -> dict[str, Path]:
    paths = {
        component: output_root / component / f"{evaluation_id}.npz"
        for component in COMPONENTS
    }
    paths["metadata"] = output_root / "metadata" / f"{evaluation_id}.json"
    return paths


def _recording_fingerprint(row: dict, cfg: dict, mode: str) -> str:
    source = Path(row["edf_path"])
    stat = source.stat()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": {
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "max_recording_sec": float(
            active_dataset_config(cfg)["max_recording_sec"]
        ),
        "preprocessing": cfg["preprocessing"],
        "standard_19": cfg["channels"]["standard_19"],
        "wiener_mode": mode,
        "wiener_config_fingerprint": make_wiener_cache_fingerprint(cfg, mode),
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _complete_cache_matches(paths: dict[str, Path], fingerprint: str) -> bool:
    expected_paths = [paths["specific"], paths["coherent"], paths["metadata"]]
    if not all(path.is_file() for path in expected_paths):
        return False

    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise CacheFingerprintMismatch(
                f"Cache schema mismatch in {paths['metadata']}"
            )
        cached_fingerprints = [metadata.get("fingerprint")]
        for component in COMPONENTS:
            with np.load(paths[component], allow_pickle=False) as data:
                required = {
                    "sequence", "component", "ch_names", "sfreq", "fingerprint",
                    "schema_version",
                }
                if not required.issubset(data.files):
                    raise CacheFingerprintMismatch(
                        f"Incomplete {component} cache: {paths[component]}"
                    )
                if str(data["component"].item()) != component:
                    raise CacheFingerprintMismatch(
                        f"Component mismatch in {paths[component]}"
                    )
                if int(data["schema_version"].item()) != CACHE_SCHEMA_VERSION:
                    raise CacheFingerprintMismatch(
                        f"Cache schema mismatch in {paths[component]}"
                    )
                cached_fingerprints.append(str(data["fingerprint"].item()))
    except CacheFingerprintMismatch:
        raise
    except Exception as exc:
        raise CacheFingerprintMismatch(
            f"Unreadable continuous cache: {exc}"
        ) from exc

    if any(cached != fingerprint for cached in cached_fingerprints):
        raise CacheFingerprintMismatch(
            "Continuous Wiener cache configuration/source mismatch; "
            "re-run with --force"
        )
    return True


def _identity_payload(row: dict) -> dict[str, np.ndarray]:
    fields = (
        "dataset_name", "patient_id", "recording_id", "evaluation_id",
        "class_name", "label", "split", "source_partition",
    )
    return {key: np.asarray(row[key]) for key in fields}


def _active_channels(diagnostics: dict) -> set[str]:
    active: set[str] = set()
    for window in diagnostics.get("window_diagnostics", []):
        active.update(str(channel) for channel in window.get("channel_sources", {}))
    return active


def _compact_diagnostics(diagnostics: dict) -> dict:
    return {
        key: _jsonable(value)
        for key, value in diagnostics.items()
        if key != "window_diagnostics"
    }


def _float32_conservation_error(
    source: np.ndarray,
    specific: np.ndarray,
    coherent: np.ndarray,
) -> float:
    max_abs_error = 0.0
    for channel_index in range(source.shape[0]):
        source_channel = source[channel_index]
        specific_channel = specific[channel_index]
        coherent_channel = coherent[channel_index]
        reconstructed = specific_channel + coherent_channel
        abs_error = np.abs(reconstructed - source_channel)
        max_abs_error = max(max_abs_error, float(np.max(abs_error)))
    return max_abs_error


def _manifest_base(row: dict, paths: dict[str, Path]) -> dict:
    return {
        "dataset_name": row["dataset_name"],
        "patient_id": row["patient_id"],
        "recording_id": row["recording_id"],
        "evaluation_id": row["evaluation_id"],
        "class_name": row["class_name"],
        "label": int(row["label"]),
        "split": row["split"],
        "source_partition": row["source_partition"],
        "edf_path": row["edf_path"],
        "specific_path": str(paths["specific"]),
        "coherent_path": str(paths["coherent"]),
        "metadata_path": str(paths["metadata"]),
    }


def _process_one(args) -> dict:
    row, cfg, output_root_str, mode, force = args
    output_root = Path(output_root_str)
    evaluation_id = str(row["evaluation_id"])
    paths = _cache_paths(output_root, evaluation_id)
    manifest = _manifest_base(row, paths)

    try:
        fingerprint = _recording_fingerprint(row, cfg, mode)
        if not force and _complete_cache_matches(paths, fingerprint):
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            return {
                **manifest,
                "status": "cached",
                "cache_hit": True,
                "fingerprint": fingerprint,
                "n_channels": metadata["n_channels"],
                "n_times": metadata["n_times"],
                "sfreq": metadata["sfreq"],
                "duration_sec": metadata["duration_sec"],
                "error": "",
            }

        data_uv, ch_names, sfreq = load_edf(Path(row["edf_path"]), cfg)
        expected_channels = list(cfg["channels"]["standard_19"])
        if ch_names != expected_channels:
            raise ValueError(
                f"Unexpected channel order: {ch_names}; expected {expected_channels}"
            )
        max_recording_sec = float(
            active_dataset_config(cfg)["max_recording_sec"]
        )
        n_times = min(data_uv.shape[1], int(max_recording_sec * sfreq))
        if n_times < 2:
            raise ValueError("Recording is too short after duration limiting")
        source_duration_sec = float(data_uv.shape[1] / sfreq)
        source_uv = np.array(data_uv[:, :n_times], dtype=np.float64, copy=True)
        del data_uv

        import mne

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(sfreq),
            ch_types=["eeg"] * len(ch_names),
        )
        source_raw = mne.io.RawArray(source_uv * 1e-6, info, verbose=False)
        specific_raw, diagnostics = wiener_continuous_raw(
            source_raw, cfg, subject_id=evaluation_id
        )
        if (
            specific_raw.ch_names != ch_names
            or specific_raw.n_times != n_times
            or not np.isclose(specific_raw.info["sfreq"], sfreq)
        ):
            raise ValueError("Continuous Wiener output is not aligned with its input")

        specific_uv = specific_raw.get_data() * 1e6
        active_channels = _active_channels(diagnostics)
        inactive_channels = [ch for ch in ch_names if ch not in active_channels]
        inactive_indices = [ch_names.index(ch) for ch in inactive_channels]
        coherent_uv = source_uv - specific_uv
        if inactive_indices:
            specific_uv[inactive_indices] = source_uv[inactive_indices]
            coherent_uv[inactive_indices] = 0.0

        if not np.isfinite(specific_uv).all() or not np.isfinite(coherent_uv).all():
            raise ValueError("Continuous Wiener output contains non-finite values")
        conservation_error_uv = float(
            np.max(np.abs(source_uv - specific_uv - coherent_uv))
        )

        source32 = source_uv.astype(np.float32)
        specific32 = specific_uv.astype(np.float32)
        coherent32 = source32 - specific32
        if inactive_indices:
            specific32[inactive_indices] = source32[inactive_indices]
            coherent32[inactive_indices] = 0.0
        float32_error_uv = _float32_conservation_error(
            source32, specific32, coherent32
        )

        common_payload = {
            "ch_names": np.asarray(ch_names, dtype="U"),
            "sfreq": np.asarray(float(sfreq)),
            "duration_sec": np.asarray(float(n_times / sfreq)),
            "fingerprint": np.asarray(fingerprint),
            "schema_version": np.asarray(CACHE_SCHEMA_VERSION),
            **_identity_payload(row),
        }
        # Metadata is the commit marker. Removing it first makes an interrupted
        # overwrite recoverable as an incomplete cache on the next run.
        paths["metadata"].unlink(missing_ok=True)
        _atomic_npz(
            paths["specific"],
            sequence=specific32,
            component=np.asarray("specific"),
            **common_payload,
        )
        _atomic_npz(
            paths["coherent"],
            sequence=coherent32,
            component=np.asarray("coherent"),
            **common_payload,
        )

        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "wiener_config_fingerprint": make_wiener_cache_fingerprint(cfg, mode),
            "mode": mode,
            "source_path": str(Path(row["edf_path"]).resolve()),
            "source_duration_sec": source_duration_sec,
            "max_recording_sec": max_recording_sec,
            "duration_sec": float(n_times / sfreq),
            "n_channels": len(ch_names),
            "n_times": n_times,
            "sfreq": float(sfreq),
            "ch_names": ch_names,
            "active_channels": sorted(active_channels),
            "inactive_channels": inactive_channels,
            "max_abs_conservation_error_uv": conservation_error_uv,
            "max_abs_float32_conservation_error_uv": float32_error_uv,
            "diagnostics": _compact_diagnostics(diagnostics),
            "identity": {
                key: _jsonable(value)
                for key, value in row.items()
                if key != "edf_path"
            },
        }
        _atomic_json(metadata, paths["metadata"])
        return {
            **manifest,
            "status": "processed",
            "cache_hit": False,
            "fingerprint": fingerprint,
            "n_channels": len(ch_names),
            "n_times": n_times,
            "sfreq": float(sfreq),
            "duration_sec": float(n_times / sfreq),
            "error": "",
        }
    except Exception as exc:
        return {
            **manifest,
            "status": "failed",
            "cache_hit": False,
            "fingerprint": "",
            "n_channels": np.nan,
            "n_times": np.nan,
            "sfreq": np.nan,
            "duration_sec": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _init_worker() -> None:
    import mne

    mne.set_log_level("ERROR")


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    if workers == 1:
        return [
            _process_one(job)
            for job in tqdm(jobs, desc="TUAB continuous Wiener")
        ]

    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=get_context("spawn"),
        initializer=_init_worker,
    ) as executor:
        futures = {executor.submit(_process_one, job): job for job in jobs}
        with tqdm(total=len(futures), desc="TUAB continuous Wiener") as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    return results


def main(
    config_path: str = "configs/tuab.yaml",
    *,
    data_dir: str | None = None,
    cache_dir: str | None = None,
    mode: str | None = None,
    workers: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    cfg = deepcopy(load_config(config_path))
    if data_dir is not None:
        cfg["paths"]["data_root"] = str(Path(data_dir).expanduser().resolve())
    if cache_dir is not None:
        cfg["paths"]["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    if active_dataset_name(cfg) != "tuab":
        raise ValueError("Script 17 requires dataset.active: tuab")

    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Wiener mode must be one of {SUPPORTED_MODES}; got {effective_mode!r}"
        )
    cfg["wiener"]["mode"] = effective_mode

    output_root = (
        Path(cfg["paths"]["cache_dir"])
        / f"tuab_continuous_wiener_{effective_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    index = assign_dataset_splits(build_recording_index(cfg), cfg)
    if index.empty:
        raise RuntimeError(
            f"No TUAB EDF recordings found below {cfg['paths']['data_root']}"
        )
    order = {
        str(evaluation_id): position
        for position, evaluation_id in enumerate(index["evaluation_id"])
    }
    jobs = [
        (row.to_dict(), cfg, str(output_root), effective_mode, force)
        for _, row in index.iterrows()
    ]
    rows = _run_jobs(jobs, workers)
    rows.sort(key=lambda row: order[str(row["evaluation_id"])])
    manifest = pd.DataFrame(rows)
    _atomic_dataframe(manifest, output_root / "manifest.csv")

    counts = manifest["status"].value_counts().to_dict()
    print(
        "Done. "
        f"processed={counts.get('processed', 0)}, "
        f"cached={counts.get('cached', 0)}, failed={counts.get('failed', 0)}."
    )
    failures = manifest[manifest["status"] == "failed"]
    if not failures.empty:
        examples = "; ".join(
            f"{row.evaluation_id}: {row.error}"
            for row in failures.head(3).itertuples()
        )
        raise RuntimeError(
            f"{len(failures)} TUAB recordings failed; see "
            f"{output_root / 'manifest.csv'}. First errors: {examples}"
        )
    _atomic_yaml(cfg, output_root / "config_resolved.yaml")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache continuous TUAB ECMAD specific/coherent sequences"
    )
    parser.add_argument("--config", default="configs/tuab.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override paths.data_root (directory containing TUAB edf/)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the base cache directory",
    )
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.config,
        data_dir=arguments.data_dir,
        cache_dir=arguments.cache_dir,
        mode=arguments.mode,
        workers=arguments.workers,
        force=arguments.force,
    )
