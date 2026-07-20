#!/usr/bin/env python3
"""Benchmark Raw, standard ICA, and Wiener preprocessing on ERP-CORE Flankers.

By default the script discovers BIDS-style ``*_task-ERN_eeg.set`` and
``*_task-LRP_eeg.set`` recordings below ``~/Data/ERP_CORE``.  Each participant
is evaluated and visualized only for the task files present in its EEG folder.
The script also supports MNE's modified one-subject FIF via
``--fif`` for backwards compatibility.  The semantic annotations in that FIF,
the bare numeric annotations in EEGLAB files, and prefixed numeric annotations
are all supported.
All three branches share filtering, resampling, response events, rejection
decisions, epoch windows, and baseline correction.  Their only branch-specific
operation is no denoising, ICA, or the repository Wiener decomposition.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import spectrogram, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import (
    CANDIDATE_BELOW_COHERENCE,
    CANDIDATE_PROCESSED,
    CANDIDATE_SOLVE_FAILED,
    decompose_epoch as decompose_epoch_frequency,
)
from eeg_bg.decomposition.wiener_phasegated import (
    decompose_epoch as decompose_epoch_phasegated,
)
from eeg_bg.decomposition.wiener_zerophase import (
    decompose_epoch as decompose_epoch_zerophase,
)


METHODS = ("raw", "standard", "wiener")
COLORS = {"raw": "#737373", "standard": "#2474B5", "wiener": "#D95F02"}
LABELS = {"raw": "Raw", "standard": "Standard ICA", "wiener": "Wiener"}

_NUMERIC_STIMULUS = {
    "stimulus/11": ("compatible", "left"),
    "stimulus/12": ("compatible", "right"),
    "stimulus/21": ("incompatible", "left"),
    "stimulus/22": ("incompatible", "right"),
}
_NUMERIC_RESPONSE = {
    "response/111": ("left", True),
    "response/112": ("left", False),
    "response/121": ("left", True),
    "response/122": ("left", False),
    "response/211": ("right", False),
    "response/212": ("right", True),
    "response/221": ("right", False),
    "response/222": ("right", True),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fif",
        type=Path,
        help="Optional single Flankers FIF path (legacy one-subject input).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="ERP-CORE root containing sub-*/eeg/*_task-{ERN,LRP}_eeg.set files.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        help="Only process this subject ID (for example sub-002); repeat as needed.",
    )
    parser.add_argument(
        "--task",
        choices=("ern", "lrp", "both"),
        default="both",
        help="Only process the selected ERP task (default: both).",
    )
    parser.add_argument(
        "--config", default="configs/erp_core_flankers.yaml", help="YAML config path"
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")
    return parser.parse_args()


def _resolve_recordings(
    data_dir: Path, fif_path: Path | None
) -> list[dict[str, Path | str]]:
    if fif_path is not None:
        resolved = fif_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ERP-CORE FIF not found: {resolved}")
        return [{"subject_id": _subject_id(resolved), "ern": resolved, "lrp": resolved}]

    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ERP-CORE data directory not found: {root}")
    recordings: dict[str, dict[str, Path | str]] = {}
    for task in ("ERN", "LRP"):
        for path in sorted(root.glob(f"sub-*/eeg/*_task-{task}_eeg.set")):
            subject_id = _subject_id(path)
            recordings.setdefault(subject_id, {"subject_id": subject_id})[
                task.lower()
            ] = path
    if not recordings:
        raise FileNotFoundError(
            f"No ERP-CORE ERN/LRP EEGLAB recordings found below {root}; expected "
            "sub-*/eeg/*_task-{ERN,LRP}_eeg.set"
        )
    return [recordings[subject_id] for subject_id in sorted(recordings)]


def _select_recordings(
    recordings: list[dict[str, Path | str]],
    subjects: list[str] | None,
    task: str,
) -> list[dict[str, Path | str]]:
    selected_tasks = ("ern", "lrp") if task == "both" else (task,)
    requested_subjects = set(subjects or [])
    available_subjects = {str(recording["subject_id"]) for recording in recordings}
    missing_subjects = sorted(requested_subjects - available_subjects)
    if missing_subjects:
        raise ValueError(
            f"Requested ERP-CORE subjects not found: {missing_subjects}; "
            f"available subjects: {sorted(available_subjects)}"
        )
    selected = []
    for recording in recordings:
        subject_id = str(recording["subject_id"])
        if requested_subjects and subject_id not in requested_subjects:
            continue
        filtered = {"subject_id": subject_id}
        filtered.update(
            {selected_task: recording[selected_task] for selected_task in selected_tasks if selected_task in recording}
        )
        if len(filtered) > 1:
            selected.append(filtered)
    if not selected:
        subject_label = sorted(requested_subjects) if requested_subjects else "all subjects"
        raise ValueError(f"No {task.upper()} recordings found for {subject_label}")
    return selected


def _subject_id(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            return parent.name
    if "Subject-" in path.stem:
        return "sub-" + path.stem.split("Subject-", 1)[1].split("_", 1)[0]
    return path.stem


def _read_recording(mne, path: Path):
    if path.suffix.lower() == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        eog_types = {
            channel: "eog"
            for channel in raw.ch_names
            if "EOG" in channel.upper()
        }
        if eog_types:
            raw.set_channel_types(eog_types, verbose=False)
        return raw
    if path.suffix.lower() == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    raise ValueError(f"Unsupported ERP-CORE recording format: {path}")


def _stimulus_details(description: str) -> tuple[str, str] | None:
    if description in {"11", "12", "21", "22"}:
        description = f"stimulus/{description}"
    if description in _NUMERIC_STIMULUS:
        return _NUMERIC_STIMULUS[description]
    if not description.startswith("stimulus/"):
        return None
    compatibility = "incompatible" if "incompatible" in description else "compatible"
    if "target_left" in description:
        return compatibility, "left"
    if "target_right" in description:
        return compatibility, "right"
    return None


def _response_details(description: str) -> tuple[str, bool | None] | None:
    if description in {"111", "112", "121", "122", "211", "212", "221", "222"}:
        description = f"response/{description}"
    if description in _NUMERIC_RESPONSE:
        return _NUMERIC_RESPONSE[description]
    if description == "response/left":
        return "left", None
    if description == "response/right":
        return "right", None
    return None


def build_response_table(
    events: np.ndarray,
    event_id: dict[str, int],
    sfreq: float,
    max_lag_sec: float = 1.5,
) -> pd.DataFrame:
    """Pair each response with the nearest preceding Flankers stimulus."""
    descriptions = {value: key for key, value in event_id.items()}
    last_stimulus: tuple[int, str, str] | None = None
    rows: list[dict] = []
    for sample, _, code in events:
        description = descriptions.get(int(code), "")
        stimulus = _stimulus_details(description)
        if stimulus is not None:
            last_stimulus = (int(sample), stimulus[0], stimulus[1])
            continue
        response = _response_details(description)
        if response is None or last_stimulus is None:
            continue
        stim_sample, compatibility, target_side = last_stimulus
        reaction_time = (int(sample) - stim_sample) / float(sfreq)
        if reaction_time < 0 or reaction_time > max_lag_sec:
            continue
        response_side, encoded_correct = response
        correct = target_side == response_side if encoded_correct is None else encoded_correct
        rows.append(
            {
                "sample": int(sample),
                "correct": bool(correct),
                "response_side": response_side,
                "target_side": target_side,
                "compatibility": compatibility,
                "reaction_time_sec": float(reaction_time),
                "response_description": description,
            }
        )
        last_stimulus = None
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(
            "No valid stimulus-response pairs found. Expected MNE semantic Flankers "
            "annotations or ERP-CORE numeric event codes."
        )
    if table["correct"].nunique() != 2:
        raise ValueError("Both correct and incorrect response trials are required for ERN.")
    return table


def _common_preprocess(raw, cfg: dict):
    raw = raw.copy().pick("eeg")
    raw.set_montage(
        "standard_1005", match_case=False, on_missing="ignore", verbose=False
    )
    line_freq = float(cfg["erp_core"]["line_freq"])
    if line_freq < raw.info["sfreq"] / 2:
        raw.notch_filter(line_freq, verbose=False)
    low, high = map(float, cfg["preprocessing"]["bandpass"])
    raw.filter(low, high, method="fir", verbose=False)
    target_sfreq = float(cfg["preprocessing"]["target_sfreq"])
    if not np.isclose(raw.info["sfreq"], target_sfreq):
        raw.resample(target_sfreq, verbose=False)
    return raw


def _standard_ica(raw, cfg: dict):
    import mne

    ica_cfg = cfg["erp_core"]["standard_ica"]
    fit_raw = raw.copy().filter(1.0, None, method="fir", verbose=False)
    n_eeg = len(mne.pick_types(fit_raw.info, eeg=True, exclude="bads"))
    n_components = min(int(ica_cfg["n_components"]), max(1, n_eeg - 1))
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=int(ica_cfg["random_state"]),
        max_iter=int(ica_cfg["max_iter"]),
        method="fastica",
    )
    ica.fit(fit_raw, picks="eeg", verbose=False)
    proxies = [ch for ch in ica_cfg["eog_proxy_channels"] if ch in raw.ch_names]
    excluded: list[int] = []
    for proxy in proxies:
        indices, _ = ica.find_bads_eog(fit_raw, ch_name=proxy, verbose=False)
        excluded.extend(indices)
    ica.exclude = sorted(set(excluded))
    return ica.apply(raw.copy(), verbose=False), ica.exclude


def _select_wiener_decomposer(mode: str):
    if mode == "frequency":
        return decompose_epoch_frequency
    if mode == "phasegated":
        return decompose_epoch_phasegated
    if mode == "zerophase":
        return decompose_epoch_zerophase
    raise ValueError(
        "ERP-CORE benchmark supports Wiener mode 'frequency', 'phasegated', "
        f"or 'zerophase', got {mode!r}"
    )


def _group_key(group: list[str]) -> str:
    """Stable group identifier matching wiener.py's ``pair_key`` construction."""
    return "-".join(group)


def _active_groups_from_result(result, group_keys: list[str]) -> set[str]:
    """Return the set of group keys with at least one accepted candidate.

    A group is considered active in a window when any of its per-target
    candidates has status ``CANDIDATE_PROCESSED``.  Candidate keys are
    formatted ``"{pair_key}::{channel}"`` (see ``wiener.py``); the channel
    suffix is redundant here because the pair key already identifies the group.
    """
    active: set[str] = set()
    keys = result.candidate_keys
    status = result.candidate_status
    if keys is not None and status is not None:
        for key, code in zip(keys, status):
            if int(code) != CANDIDATE_PROCESSED:
                continue
            pair_key = key.split("::", 1)[0]
            active.add(pair_key)
        return active
    # Fallback: derive group membership from channel_sources, which lists the
    # pair_key of each accepted candidate as the sources of a target channel.
    for sources in result.channel_sources.values():
        active.update(sources)
    return active


def _merge_group_processing_rates(
    rates_list: list[dict[str, dict]],
) -> dict[str, dict[str, int | float]]:
    """Sum ``active_windows``/``total_windows`` across multiple Wiener runs.

    Each input entry is the ``group_processing_rates`` dict produced by
    ``_wiener_continuous`` for one continuous recording (typically one ERP
    task).  Returns the per-group totals and the active-window fraction.
    """
    totals: dict[str, dict[str, int]] = {}
    for rates in rates_list:
        for group, stats in rates.items():
            entry = totals.setdefault(
                group, {"active_windows": 0, "total_windows": 0}
            )
            entry["active_windows"] += int(stats["active_windows"])
            entry["total_windows"] += int(stats["total_windows"])
    merged: dict[str, dict[str, int | float]] = {}
    for group, stats in totals.items():
        total = stats["total_windows"]
        active = stats["active_windows"]
        merged[group] = {
            "active_windows": active,
            "total_windows": total,
            "rate": float(active) / float(total) if total else 0.0,
        }
    return merged


def _wiener_continuous(raw, cfg: dict, subject_id: str = "erp_core"):
    """Apply 20 s Wiener windows with 50% weighted overlap-add."""
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    local_cfg = json.loads(json.dumps(cfg))
    local_cfg["preprocessing"]["target_sfreq"] = sfreq
    mode = str(local_cfg["wiener"].get("mode", "frequency"))
    decomposer = _select_wiener_decomposer(mode)
    n_times = max(2, int(round(cfg["preprocessing"]["epoch_length_sec"] * sfreq)))
    if data.shape[1] < n_times:
        raise ValueError("Recording is shorter than one Wiener analysis window")
    hop = n_times // 2
    pad = hop
    padded = np.pad(data, ((0, 0), (pad, pad)), mode="reflect")
    starts = list(range(0, padded.shape[1] - n_times + 1, hop))
    last = padded.shape[1] - n_times
    if starts[-1] != last:
        starts.append(last)
    window = np.hanning(n_times + 2)[1:-1]
    numerator = np.zeros_like(padded)
    denominator = np.zeros(padded.shape[1], dtype=float)
    processed = 0
    group_keys = [_group_key(group) for group in cfg["channels"]["channel_groups"]]
    group_active_windows = {key: 0 for key in group_keys}
    for epoch_idx, start in enumerate(starts):
        chunk = padded[:, start : start + n_times]
        result = decomposer(
            chunk, raw.ch_names, local_cfg, subject_id=subject_id, epoch_idx=epoch_idx
        )
        numerator[:, start : start + n_times] += result.specific * window
        denominator[start : start + n_times] += window
        processed += int(np.count_nonzero(result.channel_sources))
        for key in _active_groups_from_result(result, group_keys):
            group_active_windows[key] = group_active_windows.get(key, 0) + 1
    output = numerator / np.maximum(denominator, np.finfo(float).eps)
    denoised = raw.copy()
    denoised._data = output[:, pad : pad + data.shape[1]]
    total_windows = len(starts)
    group_processing_rates = {
        key: {
            "active_windows": group_active_windows.get(key, 0),
            "total_windows": total_windows,
            "rate": (
                float(group_active_windows.get(key, 0)) / float(total_windows)
                if total_windows
                else 0.0
            ),
        }
        for key in group_keys
    }
    return denoised, {
        "mode": mode,
        "windows": total_windows,
        "processed_channel_windows": processed,
        "group_processing_rates": group_processing_rates,
    }


def _make_shared_epochs(raws: dict, table: pd.DataFrame, spec: dict, reject_uv: float):
    import mne

    events = np.column_stack(
        [table["sample"].to_numpy(int), np.zeros(len(table), int), np.ones(len(table), int)]
    )
    metadata = table.drop(columns="sample").reset_index(drop=True)
    probe = mne.Epochs(
        raws["raw"],
        events,
        event_id={"response": 1},
        tmin=float(spec["tmin"]),
        tmax=float(spec["tmax"]),
        baseline=None,
        reject={"eeg": float(reject_uv) * 1e-6},
        metadata=metadata,
        preload=True,
        verbose=False,
    )
    selected = probe.selection
    selected_events = events[selected]
    selected_metadata = metadata.iloc[selected].reset_index(drop=True)
    baseline = tuple(float(value) for value in spec["baseline"])
    return {
        method: mne.Epochs(
            branch,
            selected_events,
            event_id={"response": 1},
            tmin=float(spec["tmin"]),
            tmax=float(spec["tmax"]),
            baseline=baseline,
            metadata=selected_metadata,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )
        for method, branch in raws.items()
    }


def _task_epoch_issue(task: str, epochs: dict) -> str | None:
    metadata = epochs["raw"].metadata
    if task == "ern":
        if metadata["correct"].nunique() != 2:
            return "shared rejection left no usable correct/incorrect ERN pair"
        return None
    correct = metadata[metadata["correct"].to_numpy(bool)]
    if set(correct["response_side"]) != {"left", "right"}:
        return "shared rejection left no usable left/right correct LRP pair"
    for compatibility in ("compatible", "incompatible"):
        subset = correct[correct["compatibility"] == compatibility]
        if set(subset["response_side"]) != {"left", "right"}:
            return (
                "shared rejection left no usable left/right correct LRP pair for "
                f"{compatibility} trials"
            )
    return None


def compute_lrp(
    epochs, compatibility: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correct-trial C3/C4 LRP mean and across-trial standard deviation."""
    correct = epochs[epochs.metadata["correct"].to_numpy(bool)]
    if compatibility is not None:
        correct = correct[
            correct.metadata["compatibility"].to_numpy() == compatibility
        ]
    left = correct[correct.metadata["response_side"].to_numpy() == "left"]
    right = correct[correct.metadata["response_side"].to_numpy() == "right"]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("LRP requires correct trials for both left and right responses")
    left_data = left.get_data(picks=["C3", "C4"], copy=False)
    right_data = right.get_data(picks=["C3", "C4"], copy=False)
    left_lateralized = left_data[:, 0] - left_data[:, 1]
    right_lateralized = right_data[:, 1] - right_data[:, 0]
    lrp = 0.5 * (
        left_lateralized.mean(axis=0) + right_lateralized.mean(axis=0)
    )
    trial_sd = np.std(
        np.concatenate([left_lateralized, right_lateralized], axis=0), axis=0
    )
    return correct.times.copy(), lrp, trial_sd


def _ern_waveforms(
    epochs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct_mask = epochs.metadata["correct"].to_numpy(bool)
    incorrect_epochs = epochs[~correct_mask]
    correct_epochs = epochs[correct_mask]
    incorrect = incorrect_epochs.get_data(picks=["FCz"], copy=False)[:, 0]
    correct = correct_epochs.get_data(picks=["FCz"], copy=False)[:, 0]
    incorrect_mean = incorrect.mean(axis=0)
    correct_mean = correct.mean(axis=0)
    difference = incorrect_mean - correct_mean
    incorrect_sd = np.std(incorrect, axis=0)
    difference_sd = np.sqrt(np.var(incorrect, axis=0) + np.var(correct, axis=0))
    return (
        incorrect_epochs.times.copy(),
        incorrect_mean,
        difference,
        incorrect_sd,
        difference_sd,
    )


def _window_mask(times: np.ndarray, limits: list[float] | tuple[float, float]) -> np.ndarray:
    return (times >= float(limits[0])) & (times <= float(limits[1]))


def _line_power(raw, freq: float) -> float:
    data = raw.get_data()
    frequencies, psd = welch(data, fs=raw.info["sfreq"], nperseg=min(1024, data.shape[1]))
    mask = (frequencies >= freq - 1.0) & (frequencies <= freq + 1.0)
    return float(np.mean(psd[:, mask])) if np.any(mask) else float("nan")


def _eog_proxy_variance_uv2(raw) -> float:
    picks = [ch for ch in ("FP1", "FP2") if ch in raw.ch_names]
    if not picks:
        return float("nan")
    proxy_uv = raw.get_data(picks=picks).mean(axis=0) * 1e6
    return float(np.var(proxy_uv))


def _half_peak_onset_ms(times: np.ndarray, wave: np.ndarray, limits) -> float:
    mask = _window_mask(times, limits)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return float("nan")
    segment = wave[indices]
    peak_local = int(np.argmax(np.abs(segment)))
    threshold = 0.5 * abs(segment[peak_local])
    crossings = np.flatnonzero(np.abs(segment[: peak_local + 1]) >= threshold)
    return float(times[indices[crossings[0]]] * 1000) if len(crossings) else float("nan")


def _classification_metrics(epochs, cfg: dict) -> dict[str, float]:
    class_cfg = cfg["erp_core"]["classification"]
    mask = _window_mask(epochs.times, [class_cfg["tmin"], class_cfg["tmax"]])
    decim = int(class_cfg["decim"])
    x = epochs.get_data(copy=False)[:, :, mask][:, :, ::decim].reshape(len(epochs), -1)
    y = (~epochs.metadata["correct"].to_numpy(bool)).astype(int)
    min_class = int(np.bincount(y).min())
    folds = min(int(class_cfg["cv_folds"]), min_class)
    if folds < 2:
        return {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")}
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(class_cfg["random_state"])
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "auc": float(roc_auc_score(y, probabilities)),
    }


def _compute_metrics(
    raws: dict,
    ern_epochs: dict | None,
    lrp_epochs: dict | None,
    cfg: dict,
):
    ern = (
        {method: _ern_waveforms(epochs) for method, epochs in ern_epochs.items()}
        if ern_epochs is not None
        else None
    )
    lrp = (
        {method: compute_lrp(epochs) for method, epochs in lrp_epochs.items()}
        if lrp_epochs is not None
        else None
    )
    reference = ern["standard"][2] if ern is not None else None
    peak_mask = (
        _window_mask(ern["standard"][0], cfg["erp_core"]["ern"]["peak_window"])
        if ern is not None
        else None
    )
    target_epochs = ern_epochs if ern_epochs is not None else lrp_epochs
    rows = []
    for method in METHODS:
        row = {
            "method": method,
            "ern_snr_db": float("nan"),
            "ern_waveform_r": float("nan"),
            "ern_rmse_vs_standard_uv": float("nan"),
            "ern_peak_uv": float("nan"),
            "ern_peak_latency_ms": float("nan"),
            "baseline_noise_sd_uv": float("nan"),
            "lrp_peak_uv": float("nan"),
            "lrp_peak_latency_ms": float("nan"),
            "lrp_half_peak_onset_ms": float("nan"),
            "line_frequency_power_v2_hz": _line_power(
                raws[method], float(cfg["erp_core"]["line_freq"])
            ),
            "fp1_fp2_proxy_variance_uv2": _eog_proxy_variance_uv2(raws[method]),
            "classification_accuracy": float("nan"),
            "classification_f1": float("nan"),
            "classification_auc": float("nan"),
        }
        if target_epochs is not None:
            raw_target = target_epochs["raw"].get_data(
                picks=["FCz", "C3", "C4"], copy=False
            )
            method_target = target_epochs[method].get_data(
                picks=["FCz", "C3", "C4"], copy=False
            )
        else:
            raw_target = raws["raw"].get_data(picks=["FCz", "C3", "C4"])
            method_target = raws[method].get_data(picks=["FCz", "C3", "C4"])
        row["target_change_rms_uv"] = float(
            np.sqrt(np.mean((method_target - raw_target) ** 2)) * 1e6
        )
        if ern is not None and ern_epochs is not None:
            times, incorrect, difference, _, _ = ern[method]
            peak_indices = np.flatnonzero(peak_mask)
            peak_index = int(peak_indices[np.argmin(difference[peak_mask])])
            baseline_mask = _window_mask(times, cfg["erp_core"]["ern"]["baseline"])
            baseline_sd = float(np.std(incorrect[baseline_mask]))
            classification = _classification_metrics(ern_epochs[method], cfg)
            row.update(
                {
                    "ern_snr_db": float(
                        20
                        * np.log10(
                            max(abs(difference[peak_index]), np.finfo(float).eps)
                            / max(baseline_sd, np.finfo(float).eps)
                        )
                    ),
                    "ern_waveform_r": float(np.corrcoef(difference, reference)[0, 1]),
                    "ern_rmse_vs_standard_uv": float(
                        np.sqrt(np.mean((difference - reference) ** 2)) * 1e6
                    ),
                    "ern_peak_uv": float(difference[peak_index] * 1e6),
                    "ern_peak_latency_ms": float(times[peak_index] * 1000),
                    "baseline_noise_sd_uv": baseline_sd * 1e6,
                    "classification_accuracy": classification["accuracy"],
                    "classification_f1": classification["f1"],
                    "classification_auc": classification["auc"],
                }
            )
        if lrp is not None:
            lrp_times, lrp_wave, _ = lrp[method]
            lrp_mask = _window_mask(
                lrp_times, cfg["erp_core"]["lrp"]["measure_window"]
            )
            lrp_peak_index = int(
                np.flatnonzero(lrp_mask)[np.argmax(np.abs(lrp_wave[lrp_mask]))]
            )
            row.update(
                {
                    "lrp_peak_uv": float(lrp_wave[lrp_peak_index] * 1e6),
                    "lrp_peak_latency_ms": float(lrp_times[lrp_peak_index] * 1000),
                    "lrp_half_peak_onset_ms": _half_peak_onset_ms(
                        lrp_times,
                        lrp_wave,
                        cfg["erp_core"]["lrp"]["measure_window"],
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), ern, lrp


def _plot_ern(
    ern: dict, cfg: dict, path: Path, show_trial_variance: bool = False
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for method in METHODS:
        times, incorrect, difference, incorrect_sd, difference_sd = ern[method]
        axes[0].plot(
            times * 1000, incorrect * 1e6, color=COLORS[method], label=LABELS[method]
        )
        axes[1].plot(
            times * 1000, difference * 1e6, color=COLORS[method], label=LABELS[method]
        )
        if show_trial_variance:
            axes[0].fill_between(
                times * 1000,
                (incorrect - incorrect_sd) * 1e6,
                (incorrect + incorrect_sd) * 1e6,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
            axes[1].fill_between(
                times * 1000,
                (difference - difference_sd) * 1e6,
                (difference + difference_sd) * 1e6,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
    limits = cfg["erp_core"]["ern"]["peak_window"]
    for ax in axes:
        ax.axvspan(limits[0] * 1000, limits[1] * 1000, color="#E8E8E8", zorder=-1)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Response-locked time (ms)")
        ax.invert_yaxis()
    axes[0].set(ylabel="Amplitude (µV)", title="FCz incorrect trials")
    axes[1].set(ylabel="Incorrect − correct (µV)", title="FCz ERN difference wave")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Wiener "
        f"mode={cfg['wiener'].get('mode', 'frequency')}, "
        f"coherence={float(cfg['wiener']['coherence_threshold']):.3f}, "
        f"phase={float(cfg['wiener']['phase_gate_threshold_rad']):.3f} rad"
        + ("\nShading: mean ±1 trial SD" if show_trial_variance else "")
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lrp(lrp: dict, path: Path, show_trial_variance: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for method in METHODS:
        times, wave, trial_sd = lrp[method]
        ax.plot(times * 1000, wave * 1e6, color=COLORS[method], label=LABELS[method])
        if show_trial_variance:
            ax.fill_between(
                times * 1000,
                (wave - trial_sd) * 1e6,
                (wave + trial_sd) * 1e6,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    title = "C3/C4 LRP"
    if show_trial_variance:
        title += " (shading: mean ±1 trial SD)"
    ax.set(xlabel="Response-locked time (ms)", ylabel="Ipsilateral − contralateral (µV)", title=title)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _lrp_compatibility_waveforms(lrp_epochs: dict) -> dict:
    return {
        compatibility: {
            method: compute_lrp(lrp_epochs[method], compatibility)
            for method in METHODS
        }
        for compatibility in ("compatible", "incompatible")
    }


def _plot_lrp_by_compatibility(
    waveforms: dict, path: Path, show_trial_variance: bool = False
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, compatibility in zip(axes, ("compatible", "incompatible")):
        for method in METHODS:
            times, wave, trial_sd = waveforms[compatibility][method]
            ax.plot(times * 1000, wave * 1e6, color=COLORS[method], label=LABELS[method])
            if show_trial_variance:
                ax.fill_between(
                    times * 1000,
                    (wave - trial_sd) * 1e6,
                    (wave + trial_sd) * 1e6,
                    color=COLORS[method],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(xlabel="Response-locked time (ms)", title=compatibility.capitalize())
    axes[0].set_ylabel("Ipsilateral − contralateral (µV)")
    axes[0].legend(frameon=False)
    title = "C3/C4 LRP by Flankers compatibility"
    if show_trial_variance:
        title += " (shading: mean ±1 trial SD)"
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_segment(raws: dict, cfg: dict, path: Path) -> None:
    vis = cfg["erp_core"]["visualization"]
    start = min(float(vis["segment_start_sec"]), raws["raw"].times[-1] - float(vis["segment_duration_sec"]))
    start = max(0.0, start)
    stop = start + float(vis["segment_duration_sec"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, method in zip(axes, METHODS):
        branch = raws[method].copy().crop(start, stop)
        picks = [ch for ch in ("FP1", "FCz", "C3", "C4") if ch in branch.ch_names]
        data = branch.get_data(picks=picks) * 1e6
        scale = max(float(np.nanpercentile(np.abs(data), 98)), 1.0)
        for index, (channel, trace) in enumerate(zip(picks, data)):
            ax.plot(branch.times, trace + index * 2.5 * scale, linewidth=0.65, label=channel)
        ax.set_title(LABELS[method], loc="left")
        ax.set_ylabel("offset µV")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_time_frequency(raws: dict, cfg: dict, path: Path) -> None:
    vis = cfg["erp_core"]["visualization"]
    start = min(
        float(vis["segment_start_sec"]),
        raws["raw"].times[-1] - float(vis["segment_duration_sec"]),
    )
    start = max(0.0, start)
    stop = start + float(vis["segment_duration_sec"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    images = []
    for ax, method in zip(axes, METHODS):
        branch = raws[method].copy().crop(start, stop)
        channel = "FP1" if "FP1" in branch.ch_names else "FCz"
        signal_uv = branch.get_data(picks=[channel])[0] * 1e6
        freqs, times, power = spectrogram(
            signal_uv,
            fs=branch.info["sfreq"],
            nperseg=min(int(branch.info["sfreq"]), len(signal_uv)),
            noverlap=min(int(branch.info["sfreq"] // 2), len(signal_uv) - 1),
        )
        mask = (freqs >= 1.0) & (freqs <= 30.0)
        db = 10 * np.log10(np.maximum(power[mask], np.finfo(float).tiny))
        images.append((ax, times + start, freqs[mask], db))
    vmin = min(float(np.percentile(item[3], 5)) for item in images)
    vmax = max(float(np.percentile(item[3], 95)) for item in images)
    mesh = None
    for method, (ax, times, freqs, db) in zip(METHODS, images):
        mesh = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set(title=LABELS[method], xlabel="Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes, shrink=0.8, label="Power (dB, µV²/Hz)")
    fig.suptitle("FP1 10 s time-frequency comparison")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_topomaps(ern_epochs: dict, ern: dict, cfg: dict, path: Path) -> None:
    import mne

    times = ern["standard"][0]
    mask = _window_mask(times, cfg["erp_core"]["ern"]["peak_window"])
    indices = np.flatnonzero(mask)
    peak_index = int(indices[np.argmin(ern["standard"][2][mask])])
    maps = []
    infos = []
    for method in METHODS:
        epochs = ern_epochs[method]
        correct = epochs.metadata["correct"].to_numpy(bool)
        difference = epochs[~correct].average().data - epochs[correct].average().data
        maps.append(difference[:, peak_index] * 1e6)
        infos.append(epochs.info)
    limit = max(float(np.max(np.abs(values))) for values in maps)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    image = None
    for ax, method, values, info in zip(axes, METHODS, maps, infos):
        image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, vlim=(-limit, limit), cmap="RdBu_r")
        ax.set_title(LABELS[method])
    fig.colorbar(image, ax=axes, shrink=0.75, label="Incorrect − correct (µV)")
    fig.suptitle(f"ERN scalp distribution at {times[peak_index] * 1000:.0f} ms")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _prepare_recording(mne, recording: Path, cfg: dict, subject_id: str) -> dict:
    original = _read_recording(mne, recording)
    common = _common_preprocess(original, cfg)
    all_events, all_event_id = mne.events_from_annotations(common, verbose=False)
    table = build_response_table(
        all_events,
        all_event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    standard, excluded = _standard_ica(common, cfg)
    wiener, wiener_diagnostics = _wiener_continuous(common, cfg, subject_id)
    return {
        "recording": recording,
        "raws": {"raw": common, "standard": standard, "wiener": wiener},
        "table": table,
        "ica_excluded_components": [int(value) for value in excluded],
        "wiener": wiener_diagnostics,
    }


def _run_subject(
    mne, recordings: dict[str, Path | str], cfg: dict, out: Path
) -> dict:
    subject_id = str(recordings["subject_id"])
    out.mkdir(parents=True, exist_ok=True)
    prepared_by_path = {}
    task_data = {}
    for task in ("ern", "lrp"):
        if task not in recordings:
            continue
        recording = Path(recordings[task])
        if recording not in prepared_by_path:
            prepared_by_path[recording] = _prepare_recording(
                mne, recording, cfg, subject_id
            )
        task_data[task] = prepared_by_path[recording]

    reject_uv = float(cfg["preprocessing"]["artifact_threshold_uv"])
    ern_epochs = (
        _make_shared_epochs(
            task_data["ern"]["raws"],
            task_data["ern"]["table"],
            cfg["erp_core"]["ern"],
            reject_uv,
        )
        if "ern" in task_data
        else None
    )
    lrp_epochs = (
        _make_shared_epochs(
            task_data["lrp"]["raws"],
            task_data["lrp"]["table"],
            cfg["erp_core"]["lrp"],
            reject_uv,
        )
        if "lrp" in task_data
        else None
    )
    epoch_counts = {
        "ern": int(len(ern_epochs["raw"])) if ern_epochs is not None else 0,
        "lrp": int(len(lrp_epochs["raw"])) if lrp_epochs is not None else 0,
    }
    epoch_issues = {
        task: _task_epoch_issue(task, epochs)
        for task, epochs in (("ern", ern_epochs), ("lrp", lrp_epochs))
        if epochs is not None
    }
    for task, issue in epoch_issues.items():
        if issue is not None:
            warnings.warn(f"{subject_id} {task.upper()} skipped: {issue}", RuntimeWarning)
    if epoch_issues.get("ern") is not None:
        ern_epochs = None
    if epoch_issues.get("lrp") is not None:
        lrp_epochs = None
    primary = task_data["ern"] if "ern" in task_data else task_data["lrp"]
    metrics, ern, lrp = _compute_metrics(
        primary["raws"], ern_epochs, lrp_epochs, cfg
    )
    lrp_by_compatibility = (
        _lrp_compatibility_waveforms(lrp_epochs)
        if lrp_epochs is not None
        else None
    )
    metrics.insert(0, "subject_id", subject_id)
    tables = []
    for prepared in prepared_by_path.values():
        tasks = [task for task, value in task_data.items() if value is prepared]
        table = prepared["table"].copy()
        table.insert(0, "task", "+".join(tasks))
        table.insert(0, "subject_id", subject_id)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    table.to_csv(out / "response_trials.csv", index=False)
    wiener_runs = [prepared["wiener"] for prepared in prepared_by_path.values()]
    summary = {
        "subject_id": subject_id,
        "input_recordings": {
            task: str(recordings[task]) for task in ("ern", "lrp") if task in recordings
        },
        "available_visualizations": [
            task
            for task, epochs in (("ern", ern_epochs), ("lrp", lrp_epochs))
            if epochs is not None
        ],
        "epoch_availability": {
            task: {
                "n_epochs_after_shared_rejection": epoch_counts[task],
                "usable": epoch_issues.get(task) is None,
                "reason": epoch_issues.get(task),
            }
            for task in task_data
        },
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "n_response_trials": int(len(table)),
        "n_correct": int(table["correct"].sum()),
        "n_incorrect": int((~table["correct"]).sum()),
        "n_ern_epochs_after_shared_rejection": (
            epoch_counts["ern"]
        ),
        "n_lrp_epochs_after_shared_rejection": (
            epoch_counts["lrp"]
        ),
        "ica_excluded_components": {
            task: task_data[task]["ica_excluded_components"] for task in task_data
        },
        "wiener": {
            "mode": str(cfg["wiener"].get("mode", "frequency")),
            "windows": sum(run["windows"] for run in wiener_runs),
            "processed_channel_windows": sum(
                run["processed_channel_windows"] for run in wiener_runs
            ),
            "group_processing_rates": _merge_group_processing_rates(
                [run.get("group_processing_rates", {}) for run in wiener_runs]
            ),
        },
        "wiener_parameters": {
            "coherence_threshold": float(cfg["wiener"]["coherence_threshold"]),
            "phase_gate_threshold_rad": float(cfg["wiener"]["phase_gate_threshold_rad"]),
        },
        "fairness": "Filtering, resampling, events, rejection, epoch windows, and baselines are shared across all branches.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    if ern is not None and ern_epochs is not None:
        _plot_ern(
            ern,
            cfg,
            out / "ern_fcz_difference.png",
            show_trial_variance=True,
        )
        _plot_topomaps(ern_epochs, ern, cfg, out / "ern_topomaps.png")
    else:
        (out / "ern_fcz_difference.png").unlink(missing_ok=True)
        (out / "ern_topomaps.png").unlink(missing_ok=True)
    if lrp is not None and lrp_by_compatibility is not None:
        _plot_lrp(lrp, out / "lrp_c3_c4.png", show_trial_variance=True)
        _plot_lrp_by_compatibility(
            lrp_by_compatibility,
            out / "lrp_by_compatibility.png",
            show_trial_variance=True,
        )
    else:
        (out / "lrp_c3_c4.png").unlink(missing_ok=True)
        (out / "lrp_by_compatibility.png").unlink(missing_ok=True)
    _plot_segment(primary["raws"], cfg, out / "time_domain_segment.png")
    _plot_time_frequency(primary["raws"], cfg, out / "time_frequency_segment.png")
    if metrics.loc[metrics["method"] == "wiener", "target_change_rms_uv"].iloc[0] < 1e-12:
        warnings.warn(
            "Wiener produced no measurable change at FCz/C3/C4; inspect coherence-gate diagnostics.",
            RuntimeWarning,
        )
    return {
        "subject_id": subject_id,
        "metrics": metrics,
        "table": table,
        "summary": summary,
        "ern": ern,
        "lrp": lrp,
        "lrp_by_compatibility": lrp_by_compatibility,
    }


def _mean_method_waveforms(waveforms_by_subject: list[dict], label: str) -> dict:
    averaged = {}
    for method in METHODS:
        waveforms = [waveforms[method] for waveforms in waveforms_by_subject]
        times = waveforms[0][0]
        if any(not np.array_equal(times, waveform[0]) for waveform in waveforms[1:]):
            raise ValueError(f"Cannot average {label}: subject time axes differ")
        averaged[method] = (times.copy(),) + tuple(
            np.mean([waveform[index] for waveform in waveforms], axis=0)
            for index in range(1, len(waveforms[0]))
        )
    return averaged


def _mean_waveforms(results: list[dict], key: str) -> dict:
    return _mean_method_waveforms([result[key] for result in results], key)


def _mean_lrp_compatibility_waveforms(results: list[dict]) -> dict:
    return {
        compatibility: _mean_method_waveforms(
            [result["lrp_by_compatibility"][compatibility] for result in results],
            f"lrp_{compatibility}",
        )
        for compatibility in ("compatible", "incompatible")
    }


def run(
    fif_path: Path | None,
    config_path: str,
    output_dir: Path | None,
    force: bool,
    data_dir: Path | None = None,
    subjects: list[str] | None = None,
    task: str = "both",
) -> Path:
    import mne

    cfg = load_config(config_path)
    configured_data_dir = Path(cfg["erp_core"]["data_dir"])
    recordings = _select_recordings(
        _resolve_recordings(data_dir or configured_data_dir, fif_path),
        subjects,
        task,
    )
    out = output_dir or Path(cfg["paths"]["results_dir"])
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {out}; pass --force to overwrite files")
    out.mkdir(parents=True, exist_ok=True)

    multiple_subjects = len(recordings) > 1
    results = []
    for index, subject_recordings in enumerate(recordings, start=1):
        subject_id = str(subject_recordings["subject_id"])
        tasks = ", ".join(task.upper() for task in ("ern", "lrp") if task in subject_recordings)
        print(f"[{index}/{len(recordings)}] Processing {subject_id}: {tasks}")
        subject_out = out / "subjects" / subject_id if multiple_subjects else out
        results.append(_run_subject(mne, subject_recordings, cfg, subject_out))

    subject_metrics = pd.concat(
        [result["metrics"] for result in results], ignore_index=True
    )
    response_trials = pd.concat(
        [result["table"] for result in results], ignore_index=True
    )
    numeric_columns = subject_metrics.select_dtypes(include=[np.number]).columns
    metrics = (
        subject_metrics.groupby("method", sort=False)[list(numeric_columns)]
        .mean()
        .reindex(METHODS)
        .reset_index()
    )
    metrics.insert(1, "n_subjects", len(results))
    metrics.insert(
        2, "n_ern_subjects", sum(result["ern"] is not None for result in results)
    )
    metrics.insert(
        3, "n_lrp_subjects", sum(result["lrp"] is not None for result in results)
    )
    metrics.to_csv(out / "metrics.csv", index=False)
    subject_metrics.to_csv(out / "subject_metrics.csv", index=False)
    response_trials.to_csv(out / "response_trials.csv", index=False)

    summary = {
        "input_data_dir": (
            None
            if fif_path is not None
            else str((data_dir or configured_data_dir).expanduser().resolve())
        ),
        "input_fif": (
            str(recordings[0]["ern"]) if fif_path is not None else None
        ),
        "input_recordings": [
            str(subject_recordings[task])
            for subject_recordings in recordings
            for task in ("ern", "lrp")
            if task in subject_recordings
            and (
                task == "ern"
                or subject_recordings[task] != subject_recordings.get("ern")
            )
        ],
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "n_subjects": len(results),
        "n_ern_subjects": sum(result["ern"] is not None for result in results),
        "n_lrp_subjects": sum(result["lrp"] is not None for result in results),
        "subject_ids": [result["subject_id"] for result in results],
        "selection": {
            "requested_subject_ids": subjects,
            "task": task,
        },
        "n_response_trials": int(len(response_trials)),
        "n_correct": int(response_trials["correct"].sum()),
        "n_incorrect": int((~response_trials["correct"]).sum()),
        "n_ern_epochs_after_shared_rejection": sum(
            result["summary"]["n_ern_epochs_after_shared_rejection"]
            for result in results
        ),
        "n_lrp_epochs_after_shared_rejection": sum(
            result["summary"]["n_lrp_epochs_after_shared_rejection"]
            for result in results
        ),
        "ica_excluded_components": {
            result["subject_id"]: result["summary"]["ica_excluded_components"]
            for result in results
        },
        "wiener": {
            "mode": str(cfg["wiener"].get("mode", "frequency")),
            "windows": sum(
                result["summary"]["wiener"]["windows"] for result in results
            ),
            "processed_channel_windows": sum(
                result["summary"]["wiener"]["processed_channel_windows"]
                for result in results
            ),
            "group_processing_rates": _merge_group_processing_rates(
                [
                    result["summary"]["wiener"].get("group_processing_rates", {})
                    for result in results
                ]
            ),
        },
        "subjects": [result["summary"] for result in results],
        "wiener_parameters": {
            "coherence_threshold": float(cfg["wiener"]["coherence_threshold"]),
            "phase_gate_threshold_rad": float(cfg["wiener"]["phase_gate_threshold_rad"]),
        },
        "statistical_note": "Group metrics are equal-weight means of the available participants; this local ERP-CORE subset is not the complete release.",
        "fairness": "Each participant uses shared filtering, resampling, events, rejection, epoch windows, and baselines across all three branches.",
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    if multiple_subjects:
        ern_results = [result for result in results if result["ern"] is not None]
        lrp_results = [result for result in results if result["lrp"] is not None]
        if ern_results:
            _plot_ern(
                _mean_waveforms(ern_results, "ern"),
                cfg,
                out / "ern_fcz_difference.png",
            )
        else:
            (out / "ern_fcz_difference.png").unlink(missing_ok=True)
        if lrp_results:
            _plot_lrp(
                _mean_waveforms(lrp_results, "lrp"), out / "lrp_c3_c4.png"
            )
            _plot_lrp_by_compatibility(
                _mean_lrp_compatibility_waveforms(lrp_results),
                out / "lrp_by_compatibility.png",
            )
        else:
            (out / "lrp_c3_c4.png").unlink(missing_ok=True)
            (out / "lrp_by_compatibility.png").unlink(missing_ok=True)
        for stale_single_subject_plot in (
            "ern_topomaps.png",
            "time_domain_segment.png",
            "time_frequency_segment.png",
        ):
            (out / stale_single_subject_plot).unlink(missing_ok=True)
    return out


def main() -> None:
    args = _parse_args()
    output = run(
        args.fif,
        args.config,
        args.output_dir,
        args.force,
        data_dir=args.data_dir,
        subjects=args.subjects,
        task=args.task,
    )
    print(f"ERP-CORE benchmark complete: {output}")


if __name__ == "__main__":
    main()
