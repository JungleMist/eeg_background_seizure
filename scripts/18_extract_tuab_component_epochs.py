"""Extract paired raw/specific/coherent TUAB component epochs."""
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
from eeg_bg.io.dataset import active_dataset_name


CACHE_SCHEMA_VERSION = 2
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
COMPONENTS = ("specific", "coherent")
COMBINED_CONDITION = "specific_coherent"
IDENTITY_FIELDS = (
    "patient_id", "recording_id", "evaluation_id", "class_name", "label",
    "split", "source_partition",
)


class CacheMismatch(ValueError):
    """Raised when an existing cache does not match the requested inputs."""


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
            yaml.safe_dump(_jsonable(payload), sort_keys=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_paths(input_root: Path, evaluation_id: str) -> dict[str, Path]:
    return {
        "specific": input_root / "specific" / f"{evaluation_id}.npz",
        "coherent": input_root / "coherent" / f"{evaluation_id}.npz",
        "metadata": input_root / "metadata" / f"{evaluation_id}.json",
    }


def discover_inputs(input_root: Path) -> list[dict[str, Path]]:
    required_dirs = [input_root / name for name in (*COMPONENTS, "metadata")]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Script 17 cache directories not found: " + ", ".join(missing_dirs)
        )
    stems = set()
    for component in COMPONENTS:
        stems.update(path.stem for path in (input_root / component).glob("*.npz"))
    stems.update(path.stem for path in (input_root / "metadata").glob("*.json"))
    if not stems:
        raise FileNotFoundError(f"No Script 17 cache files found in {input_root}")
    return [_input_paths(input_root, stem) for stem in sorted(stems)]


def _scalar(data, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _combined_channel_names(ch_names: list[str] | tuple[str, ...]) -> list[str]:
    return [
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    ]


def _read_header(paths: dict[str, Path], cfg: dict) -> dict:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete Script 17 cache: " + ", ".join(missing))

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported Script 17 schema: {paths['metadata']}")
    metadata_fingerprint = str(metadata.get("fingerprint", ""))
    headers: dict[str, dict] = {}
    required = {
        "sequence", "component", "ch_names", "sfreq", "fingerprint",
        "schema_version", *IDENTITY_FIELDS,
    }
    for component in COMPONENTS:
        with np.load(paths[component], allow_pickle=False) as data:
            if not required.issubset(data.files):
                absent = sorted(required - set(data.files))
                raise ValueError(f"Missing keys in {paths[component]}: {absent}")
            header = {key: _scalar(data, key) for key in IDENTITY_FIELDS}
            header.update({
                "component": str(_scalar(data, "component")),
                "ch_names": [str(item) for item in data["ch_names"]],
                "sfreq": float(_scalar(data, "sfreq")),
                "fingerprint": str(_scalar(data, "fingerprint")),
                "schema_version": int(_scalar(data, "schema_version")),
            })
            headers[component] = header

    specific = headers["specific"]
    coherent = headers["coherent"]
    for component, header in headers.items():
        if header["component"] != component:
            raise ValueError(f"Component mismatch in {paths[component]}")
        if header["schema_version"] != 1:
            raise ValueError(f"Unsupported Script 17 schema in {paths[component]}")
        if header["fingerprint"] != metadata_fingerprint:
            raise ValueError(f"Fingerprint mismatch in {paths[component]}")
    for field in IDENTITY_FIELDS:
        if str(specific[field]) != str(coherent[field]):
            raise ValueError(f"Component identity mismatch for {field}")
    if specific["ch_names"] != coherent["ch_names"]:
        raise ValueError("Component channel order mismatch")
    if not np.isclose(specific["sfreq"], coherent["sfreq"]):
        raise ValueError("Component sampling-rate mismatch")
    if specific["ch_names"] != list(cfg["channels"]["standard_19"]):
        raise ValueError("Component channels do not match channels.standard_19")
    if metadata.get("ch_names") != specific["ch_names"]:
        raise ValueError("Component channels do not match Script 17 metadata")
    if not np.isclose(float(metadata.get("sfreq", np.nan)), specific["sfreq"]):
        raise ValueError("Component sampling rate does not match Script 17 metadata")

    label = int(specific["label"])
    class_name = str(specific["class_name"])
    expected_class = {0: "abnormal", 1: "normal"}.get(label)
    if expected_class is None or class_name != expected_class:
        raise ValueError(
            f"Invalid TUAB label mapping: label={label}, class_name={class_name!r}"
        )
    split = str(specific["split"])
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid TUAB split: {split!r}")
    source_mode = str(metadata.get("mode", ""))
    if source_mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid Script 17 Wiener mode: {source_mode!r}")
    return {
        **specific,
        "label": label,
        "class_name": class_name,
        "split": split,
        "source_fingerprint_specific": specific["fingerprint"],
        "source_fingerprint_coherent": coherent["fingerprint"],
        "source_mode": source_mode,
        "source_n_times": int(metadata["n_times"]),
    }


def _epoch_fingerprint(header: dict, cfg: dict) -> str:
    preprocessing = cfg["preprocessing"]
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint_specific": header["source_fingerprint_specific"],
        "source_fingerprint_coherent": header["source_fingerprint_coherent"],
        "epoch_length_sec": float(preprocessing["epoch_length_sec"]),
        "artifact_threshold_uv": float(preprocessing["artifact_threshold_uv"]),
        "rejection_policy": "raw_shared_mask",
        "ch_names": header["ch_names"],
        "specific_coherent_ch_names": _combined_channel_names(header["ch_names"]),
        "sfreq": float(header["sfreq"]),
        "identity": {key: _jsonable(header[key]) for key in IDENTITY_FIELDS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _output_cache_matches(path: Path, fingerprint: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "raw", "specific", "coherent", COMBINED_CONDITION,
                "ch_names", "specific_coherent_ch_names",
                "fingerprint", "schema_version",
                "n_candidate_epochs", "n_rejected_epochs", "n_epochs",
                "epoch_samples", "sfreq",
            }
            if not required.issubset(data.files):
                raise CacheMismatch(f"Incomplete epoch cache: {path}")
            if int(_scalar(data, "schema_version")) != CACHE_SCHEMA_VERSION:
                raise CacheMismatch(f"Epoch cache schema mismatch: {path}")
            if str(_scalar(data, "fingerprint")) != fingerprint:
                raise CacheMismatch(
                    f"Epoch cache configuration/source mismatch: {path}; "
                    "re-run with --force"
                )
            n_epochs = int(_scalar(data, "n_epochs"))
            epoch_samples = int(_scalar(data, "epoch_samples"))
            ch_names = [str(value) for value in data["ch_names"]]
            expected_shape = (n_epochs, len(ch_names), epoch_samples)
            combined_shape = (n_epochs, 2 * len(ch_names), epoch_samples)
            for condition in ("raw", "specific", "coherent"):
                if data[condition].shape != expected_shape:
                    raise CacheMismatch(
                        f"Epoch cache {condition} shape mismatch: {path}"
                    )
                if data[condition].dtype != np.dtype(np.float32):
                    raise CacheMismatch(
                        f"Epoch cache {condition} dtype mismatch: {path}"
                    )
            if data[COMBINED_CONDITION].shape != combined_shape:
                raise CacheMismatch(
                    f"Epoch cache {COMBINED_CONDITION} shape mismatch: {path}"
                )
            if data[COMBINED_CONDITION].dtype != np.dtype(np.float32):
                raise CacheMismatch(
                    f"Epoch cache {COMBINED_CONDITION} dtype mismatch: {path}"
                )
            combined_names = [
                str(value) for value in data["specific_coherent_ch_names"]
            ]
            if combined_names != _combined_channel_names(ch_names):
                raise CacheMismatch(
                    f"Epoch cache combined channel order mismatch: {path}"
                )
            return {
                "n_candidate_epochs": int(_scalar(data, "n_candidate_epochs")),
                "n_rejected_epochs": int(_scalar(data, "n_rejected_epochs")),
                "n_epochs": int(_scalar(data, "n_epochs")),
                "epoch_samples": int(_scalar(data, "epoch_samples")),
                "sfreq": float(_scalar(data, "sfreq")),
            }
    except CacheMismatch:
        raise
    except Exception as exc:
        raise CacheMismatch(f"Unreadable epoch cache {path}: {exc}") from exc


def _load_sequence(path: Path, expected_component: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if str(_scalar(data, "component")) != expected_component:
            raise ValueError(f"Component mismatch in {path}")
        sequence = np.asarray(data["sequence"], dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2-D continuous sequence in {path}")
    return sequence


def _extract_epochs(
    specific: np.ndarray,
    coherent: np.ndarray,
    sfreq: float,
    epoch_length_sec: float,
    artifact_threshold_uv: float,
) -> dict:
    if specific.shape != coherent.shape:
        raise ValueError("Specific/coherent sequence shape mismatch")
    if not np.isfinite(specific).all() or not np.isfinite(coherent).all():
        raise ValueError("Source component contains non-finite values")
    raw = specific + coherent
    if not np.isfinite(raw).all():
        raise ValueError("Reconstructed raw contains non-finite values")

    epoch_samples = int(epoch_length_sec * sfreq)
    if epoch_samples < 1:
        raise ValueError("Configured epoch length is shorter than one sample")
    starts = np.arange(
        0, raw.shape[1] - epoch_samples + 1, epoch_samples, dtype=np.int64
    )
    accepted = np.asarray([
        start
        for start in starts
        if np.max(np.abs(raw[:, start:start + epoch_samples]))
        <= artifact_threshold_uv
    ], dtype=np.int64)
    shape = (0, raw.shape[0], epoch_samples)
    if accepted.size == 0:
        raw_epochs = np.empty(shape, dtype=np.float32)
        specific_epochs = np.empty(shape, dtype=np.float32)
        coherent_epochs = np.empty(shape, dtype=np.float32)
    else:
        raw_epochs = np.stack(
            [raw[:, start:start + epoch_samples] for start in accepted]
        ).astype(np.float32, copy=False)
        specific_epochs = np.stack(
            [specific[:, start:start + epoch_samples] for start in accepted]
        ).astype(np.float32, copy=False)
        coherent_epochs = np.stack(
            [coherent[:, start:start + epoch_samples] for start in accepted]
        ).astype(np.float32, copy=False)
    specific_coherent_epochs = np.concatenate(
        [specific_epochs, coherent_epochs], axis=1
    ).astype(np.float32, copy=False)
    np.testing.assert_allclose(
        specific_epochs + coherent_epochs, raw_epochs, rtol=1e-6, atol=1e-5
    )
    return {
        "raw": raw_epochs,
        "specific": specific_epochs,
        "coherent": coherent_epochs,
        COMBINED_CONDITION: specific_coherent_epochs,
        "epoch_start_samples": accepted,
        "epoch_start_sec": accepted.astype(np.float64) / sfreq,
        "epoch_samples": epoch_samples,
        "n_candidate_epochs": int(starts.size),
        "n_rejected_epochs": int(starts.size - accepted.size),
        "tail_samples": int(raw.shape[1] - starts.size * epoch_samples),
    }


def _manifest_base(paths: dict[str, Path], output_path: Path) -> dict:
    evaluation_id = paths["specific"].stem
    return {
        "evaluation_id": evaluation_id,
        "specific_path": str(paths["specific"]),
        "coherent_path": str(paths["coherent"]),
        "metadata_path": str(paths["metadata"]),
        "output_path": str(output_path),
    }


def _process_one(args) -> dict:
    paths, output_root_str, cfg, force = args
    output_path = Path(output_root_str) / f"{paths['specific'].stem}.npz"
    manifest = _manifest_base(paths, output_path)
    header: dict = {}
    try:
        header = _read_header(paths, cfg)
        fingerprint = _epoch_fingerprint(header, cfg)
        if not force:
            cached = _output_cache_matches(output_path, fingerprint)
            if cached is not None:
                return {
                    **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
                    **cached, "tail_samples": max(
                        0, header["source_n_times"]
                        - cached["n_candidate_epochs"] * cached["epoch_samples"]
                    ),
                    "status": "cached", "cache_hit": True,
                    "fingerprint": fingerprint,
                    "source_mode": header["source_mode"], "error": "",
                }

        specific = _load_sequence(paths["specific"], "specific")
        coherent = _load_sequence(paths["coherent"], "coherent")
        if specific.shape[0] != len(header["ch_names"]):
            raise ValueError("Source channel count does not match ch_names")
        if specific.shape[1] != header["source_n_times"]:
            raise ValueError("Source length does not match Script 17 metadata")
        extracted = _extract_epochs(
            specific, coherent, float(header["sfreq"]),
            float(cfg["preprocessing"]["epoch_length_sec"]),
            float(cfg["preprocessing"]["artifact_threshold_uv"]),
        )
        if extracted["raw"].shape[0] == 0:
            output_path.unlink(missing_ok=True)
            return {
                **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
                "status": "skipped", "cache_hit": False,
                "fingerprint": fingerprint,
                "n_candidate_epochs": extracted["n_candidate_epochs"],
                "n_rejected_epochs": extracted["n_rejected_epochs"],
                "n_epochs": 0, "epoch_samples": extracted["epoch_samples"],
                "tail_samples": extracted["tail_samples"],
                "sfreq": float(header["sfreq"]),
                "source_mode": header["source_mode"],
                "error": "no valid epochs",
            }

        _atomic_npz(
            output_path,
            raw=extracted["raw"], specific=extracted["specific"],
            coherent=extracted["coherent"],
            specific_coherent=extracted[COMBINED_CONDITION],
            epoch_start_samples=extracted["epoch_start_samples"],
            epoch_start_sec=extracted["epoch_start_sec"],
            label=np.asarray(int(header["label"]), dtype=np.int8),
            class_name=np.asarray(str(header["class_name"])),
            split=np.asarray(str(header["split"])),
            patient_id=np.asarray(str(header["patient_id"])),
            recording_id=np.asarray(str(header["recording_id"])),
            evaluation_id=np.asarray(str(header["evaluation_id"])),
            subject_id=np.asarray(str(header["evaluation_id"])),
            source_partition=np.asarray(str(header["source_partition"])),
            ch_names=np.asarray(header["ch_names"], dtype="U"),
            specific_coherent_ch_names=np.asarray(
                _combined_channel_names(header["ch_names"]), dtype="U"
            ),
            sfreq=np.asarray(float(header["sfreq"])),
            epoch_length_sec=np.asarray(
                float(cfg["preprocessing"]["epoch_length_sec"])
            ),
            epoch_samples=np.asarray(extracted["epoch_samples"]),
            artifact_threshold_uv=np.asarray(
                float(cfg["preprocessing"]["artifact_threshold_uv"])
            ),
            rejection_policy=np.asarray("raw_shared_mask"),
            n_candidate_epochs=np.asarray(extracted["n_candidate_epochs"]),
            n_rejected_epochs=np.asarray(extracted["n_rejected_epochs"]),
            n_epochs=np.asarray(extracted["raw"].shape[0]),
            tail_samples=np.asarray(extracted["tail_samples"]),
            source_fingerprint_specific=np.asarray(
                header["source_fingerprint_specific"]
            ),
            source_fingerprint_coherent=np.asarray(
                header["source_fingerprint_coherent"]
            ),
            source_mode=np.asarray(header["source_mode"]),
            fingerprint=np.asarray(fingerprint),
            schema_version=np.asarray(CACHE_SCHEMA_VERSION),
        )
        return {
            **manifest, **{key: header[key] for key in IDENTITY_FIELDS},
            "status": "processed", "cache_hit": False,
            "fingerprint": fingerprint,
            "n_candidate_epochs": extracted["n_candidate_epochs"],
            "n_rejected_epochs": extracted["n_rejected_epochs"],
            "n_epochs": extracted["raw"].shape[0],
            "epoch_samples": extracted["epoch_samples"],
            "tail_samples": extracted["tail_samples"],
            "sfreq": float(header["sfreq"]),
            "source_mode": header["source_mode"], "error": "",
        }
    except Exception as exc:
        identity = {key: header.get(key, "") for key in IDENTITY_FIELDS}
        identity["evaluation_id"] = identity["evaluation_id"] or manifest[
            "evaluation_id"
        ]
        return {
            **manifest, **identity, "status": "failed", "cache_hit": False,
            "fingerprint": "", "n_candidate_epochs": np.nan,
            "n_rejected_epochs": np.nan, "n_epochs": np.nan,
            "epoch_samples": np.nan, "tail_samples": np.nan, "sfreq": np.nan,
            "source_mode": header.get("source_mode", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    if workers == 1:
        return [
            _process_one(job)
            for job in tqdm(jobs, desc="TUAB paired epochs")
        ]
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        with tqdm(total=len(futures), desc="TUAB paired epochs") as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    return results


def main(
    config_path: str = "configs/tuab.yaml",
    *, input_dir: str | None = None, output_dir: str | None = None,
    mode: str | None = None, workers: int = 1, force: bool = False,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    cfg = deepcopy(load_config(config_path))
    if active_dataset_name(cfg) != "tuab":
        raise ValueError("Script 18 requires dataset.active: tuab")
    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Wiener mode must be one of {SUPPORTED_MODES}; got {effective_mode!r}"
        )
    input_root = (
        Path(input_dir).expanduser().resolve() if input_dir else
        Path(cfg["paths"]["cache_dir"]) / f"tuab_continuous_wiener_{effective_mode}"
    )
    output_root = (
        Path(output_dir).expanduser().resolve() if output_dir else input_root / "epochs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = discover_inputs(input_root)
    jobs = [(paths, str(output_root), cfg, force) for paths in inputs]
    rows = _run_jobs(jobs, workers)
    rows.sort(key=lambda row: str(row["evaluation_id"]))
    manifest = pd.DataFrame(rows)
    _atomic_dataframe(manifest, output_root / "manifest.csv")

    counts = manifest["status"].value_counts().to_dict()
    print(
        "Done. "
        f"processed={counts.get('processed', 0)}, cached={counts.get('cached', 0)}, "
        f"skipped={counts.get('skipped', 0)}, failed={counts.get('failed', 0)}."
    )
    failures = manifest[manifest["status"] == "failed"]
    if not failures.empty:
        examples = "; ".join(
            f"{row.evaluation_id}: {row.error}"
            for row in failures.head(3).itertuples()
        )
        raise RuntimeError(
            f"{len(failures)} TUAB epoch caches failed; see "
            f"{output_root / 'manifest.csv'}. First errors: {examples}"
        )
    resolved = deepcopy(cfg)
    resolved["tuab_component_epochs"] = {
        "input_dir": str(input_root), "output_dir": str(output_root),
        "requested_mode": effective_mode,
        "source_modes": sorted(set(manifest["source_mode"].astype(str))),
        "rejection_policy": "raw_shared_mask",
        "conditions": ["raw", "specific", "coherent", COMBINED_CONDITION],
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
    }
    _atomic_yaml(resolved, output_root / "config_resolved.yaml")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract paired TUAB raw/specific/coherent/specific_coherent "
            "epoch caches"
        )
    )
    parser.add_argument("--config", default="configs/tuab.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.config, input_dir=arguments.input_dir,
        output_dir=arguments.output_dir, mode=arguments.mode,
        workers=arguments.workers, force=arguments.force,
    )
