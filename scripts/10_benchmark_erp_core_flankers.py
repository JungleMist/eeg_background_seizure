#!/usr/bin/env python3
"""Benchmark Raw, standard ICA, and Wiener preprocessing on ERP-CORE Flankers.

The script supports both the semantic annotations in MNE's modified one-subject
FIF and the numeric response codes used by the full ERP-CORE Flankers release.
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
from eeg_bg.decomposition.wiener import decompose_epoch as decompose_epoch_frequency
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
        help="Flankers FIF path. If omitted, download/locate it via MNE.",
    )
    parser.add_argument(
        "--config", default="configs/erp_core_flankers.yaml", help="YAML config path"
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")
    return parser.parse_args()


def _resolve_fif(mne, path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ERP-CORE FIF not found: {resolved}")
        return resolved
    data_dir = Path(mne.datasets.erp_core.data_path())
    candidate = data_dir / "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
    if not candidate.is_file():
        matches = sorted(data_dir.rglob("*Flankers*.fif"))
        if not matches:
            raise FileNotFoundError(f"No Flankers FIF found below {data_dir}")
        candidate = matches[0]
    return candidate


def _stimulus_details(description: str) -> tuple[str, str] | None:
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


def _wiener_continuous(raw, cfg: dict):
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
    for epoch_idx, start in enumerate(starts):
        chunk = padded[:, start : start + n_times]
        result = decomposer(
            chunk, raw.ch_names, local_cfg, subject_id="erp_core", epoch_idx=epoch_idx
        )
        numerator[:, start : start + n_times] += result.specific * window
        denominator[start : start + n_times] += window
        processed += int(np.count_nonzero(result.channel_sources))
    output = numerator / np.maximum(denominator, np.finfo(float).eps)
    denoised = raw.copy()
    denoised._data = output[:, pad : pad + data.shape[1]]
    return denoised, {
        "mode": mode,
        "windows": len(starts),
        "processed_channel_windows": processed,
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


def compute_lrp(epochs, compatibility: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return correct-trial ipsilateral-minus-contralateral C3/C4 LRP."""
    correct = epochs[epochs.metadata["correct"].to_numpy(bool)]
    if compatibility is not None:
        correct = correct[
            correct.metadata["compatibility"].to_numpy() == compatibility
        ]
    left = correct[correct.metadata["response_side"].to_numpy() == "left"]
    right = correct[correct.metadata["response_side"].to_numpy() == "right"]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("LRP requires correct trials for both left and right responses")
    left_data = left.get_data(picks=["C3", "C4"], copy=False).mean(axis=0)
    right_data = right.get_data(picks=["C3", "C4"], copy=False).mean(axis=0)
    lrp = 0.5 * ((left_data[0] - left_data[1]) + (right_data[1] - right_data[0]))
    return correct.times.copy(), lrp


def _ern_waveforms(epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct_mask = epochs.metadata["correct"].to_numpy(bool)
    incorrect = epochs[~correct_mask].average(picks=["FCz"])
    correct = epochs[correct_mask].average(picks=["FCz"])
    return incorrect.times.copy(), incorrect.data[0], incorrect.data[0] - correct.data[0]


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


def _compute_metrics(raws: dict, ern_epochs: dict, lrp_epochs: dict, cfg: dict):
    ern = {method: _ern_waveforms(epochs) for method, epochs in ern_epochs.items()}
    lrp = {method: compute_lrp(epochs) for method, epochs in lrp_epochs.items()}
    reference = ern["standard"][2]
    peak_mask = _window_mask(ern["standard"][0], cfg["erp_core"]["ern"]["peak_window"])
    rows = []
    for method in METHODS:
        times, incorrect, difference = ern[method]
        peak_indices = np.flatnonzero(peak_mask)
        peak_index = int(peak_indices[np.argmin(difference[peak_mask])])
        baseline_mask = _window_mask(times, cfg["erp_core"]["ern"]["baseline"])
        baseline_sd = float(np.std(incorrect[baseline_mask]))
        peak_uv = float(difference[peak_index] * 1e6)
        snr = 20 * np.log10(max(abs(difference[peak_index]), np.finfo(float).eps) / max(baseline_sd, np.finfo(float).eps))
        corr = float(np.corrcoef(difference, reference)[0, 1])
        rmse_uv = float(np.sqrt(np.mean((difference - reference) ** 2)) * 1e6)
        lrp_times, lrp_wave = lrp[method]
        lrp_mask = _window_mask(lrp_times, cfg["erp_core"]["lrp"]["measure_window"])
        lrp_peak_index = int(np.flatnonzero(lrp_mask)[np.argmax(np.abs(lrp_wave[lrp_mask]))])
        classification = _classification_metrics(ern_epochs[method], cfg)
        raw_target = ern_epochs["raw"].get_data(picks=["FCz", "C3", "C4"], copy=False)
        method_target = ern_epochs[method].get_data(picks=["FCz", "C3", "C4"], copy=False)
        rows.append(
            {
                "method": method,
                "ern_snr_db": float(snr),
                "ern_waveform_r": corr,
                "ern_rmse_vs_standard_uv": rmse_uv,
                "ern_peak_uv": peak_uv,
                "ern_peak_latency_ms": float(times[peak_index] * 1000),
                "baseline_noise_sd_uv": baseline_sd * 1e6,
                "lrp_peak_uv": float(lrp_wave[lrp_peak_index] * 1e6),
                "lrp_peak_latency_ms": float(lrp_times[lrp_peak_index] * 1000),
                "lrp_half_peak_onset_ms": _half_peak_onset_ms(
                    lrp_times, lrp_wave, cfg["erp_core"]["lrp"]["measure_window"]
                ),
                "line_50hz_power_v2_hz": _line_power(raws[method], 50.0),
                "fp1_fp2_proxy_variance_uv2": _eog_proxy_variance_uv2(raws[method]),
                "target_change_rms_uv": float(np.sqrt(np.mean((method_target - raw_target) ** 2)) * 1e6),
                "classification_accuracy": classification["accuracy"],
                "classification_f1": classification["f1"],
                "classification_auc": classification["auc"],
            }
        )
    return pd.DataFrame(rows), ern, lrp


def _plot_ern(ern: dict, cfg: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for method in METHODS:
        times, _, difference = ern[method]
        axes[0].plot(
            times * 1000, ern[method][1] * 1e6, color=COLORS[method], label=LABELS[method]
        )
        axes[1].plot(
            times * 1000, difference * 1e6, color=COLORS[method], label=LABELS[method]
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
        f"coherence={float(cfg['wiener']['coherence_threshold']):.2f}, "
        f"phase={float(cfg['wiener']['phase_gate_threshold_rad']):.3f} rad"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lrp(lrp: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for method in METHODS:
        times, wave = lrp[method]
        ax.plot(times * 1000, wave * 1e6, color=COLORS[method], label=LABELS[method])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Response-locked time (ms)", ylabel="Ipsilateral − contralateral (µV)", title="C3/C4 LRP")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lrp_by_compatibility(lrp_epochs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, compatibility in zip(axes, ("compatible", "incompatible")):
        for method in METHODS:
            times, wave = compute_lrp(lrp_epochs[method], compatibility)
            ax.plot(times * 1000, wave * 1e6, color=COLORS[method], label=LABELS[method])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(xlabel="Response-locked time (ms)", title=compatibility.capitalize())
    axes[0].set_ylabel("Ipsilateral − contralateral (µV)")
    axes[0].legend(frameon=False)
    fig.suptitle("C3/C4 LRP by Flankers compatibility")
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


def run(fif_path: Path | None, config_path: str, output_dir: Path | None, force: bool) -> Path:
    import mne

    cfg = load_config(config_path)
    fif = _resolve_fif(mne, fif_path)
    out = output_dir or Path(cfg["paths"]["results_dir"])
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {out}; pass --force to overwrite files")
    out.mkdir(parents=True, exist_ok=True)

    original = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    common = _common_preprocess(original, cfg)
    all_events, all_event_id = mne.events_from_annotations(common, verbose=False)
    table = build_response_table(
        all_events,
        all_event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    standard, excluded = _standard_ica(common, cfg)
    wiener, wiener_diagnostics = _wiener_continuous(common, cfg)
    raws = {"raw": common, "standard": standard, "wiener": wiener}

    reject_uv = float(cfg["preprocessing"]["artifact_threshold_uv"])
    ern_epochs = _make_shared_epochs(raws, table, cfg["erp_core"]["ern"], reject_uv)
    lrp_epochs = _make_shared_epochs(raws, table, cfg["erp_core"]["lrp"], reject_uv)
    metrics, ern, lrp = _compute_metrics(raws, ern_epochs, lrp_epochs, cfg)
    metrics.to_csv(out / "metrics.csv", index=False)
    table.to_csv(out / "response_trials.csv", index=False)
    summary = {
        "input_fif": str(fif),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "n_response_trials": int(len(table)),
        "n_correct": int(table["correct"].sum()),
        "n_incorrect": int((~table["correct"]).sum()),
        "n_ern_epochs_after_shared_rejection": int(len(ern_epochs["raw"])),
        "n_lrp_epochs_after_shared_rejection": int(len(lrp_epochs["raw"])),
        "ica_excluded_components": [int(value) for value in excluded],
        "wiener": wiener_diagnostics,
        "wiener_parameters": {
            "coherence_threshold": float(cfg["wiener"]["coherence_threshold"]),
            "phase_gate_threshold_rad": float(cfg["wiener"]["phase_gate_threshold_rad"]),
        },
        "statistical_note": "MNE's bundled ERP-CORE FIF contains one participant; subject-level paired t-tests require the full multi-participant release.",
        "fairness": "Filtering, resampling, events, rejection, epoch windows, and baselines are shared across all branches.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    _plot_ern(ern, cfg, out / "ern_fcz_difference.png")
    _plot_lrp(lrp, out / "lrp_c3_c4.png")
    _plot_lrp_by_compatibility(lrp_epochs, out / "lrp_by_compatibility.png")
    _plot_segment(raws, cfg, out / "time_domain_segment.png")
    _plot_time_frequency(raws, cfg, out / "time_frequency_segment.png")
    _plot_topomaps(ern_epochs, ern, cfg, out / "ern_topomaps.png")
    if metrics.loc[metrics["method"] == "wiener", "target_change_rms_uv"].iloc[0] < 1e-12:
        warnings.warn(
            "Wiener produced no measurable change at FCz/C3/C4; inspect coherence-gate diagnostics.",
            RuntimeWarning,
        )
    return out


def main() -> None:
    args = _parse_args()
    output = run(args.fif, args.config, args.output_dir, args.force)
    print(f"ERP-CORE benchmark complete: {output}")


if __name__ == "__main__":
    main()
