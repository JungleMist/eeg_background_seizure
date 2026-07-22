from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import mne
import numpy as np

from eeg_bg.config.settings import load_config
from eeg_bg.exceptions import ProcessingCancelled
from eeg_bg.preprocessing.continuous import wiener_continuous_raw

from .models import ProcessingSpec
from .recording import RecordingService


METHODS = ("raw", "ica", "wiener")

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


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ResponseTrial:
    sample: int
    correct: bool
    response_side: str
    target_side: str
    compatibility: str
    reaction_time_sec: float


@dataclass(frozen=True)
class ErnWaveform:
    times_ms: np.ndarray
    incorrect_mean_uv: np.ndarray
    difference_mean_uv: np.ndarray
    incorrect_sd_uv: np.ndarray
    difference_sd_uv: np.ndarray


@dataclass(frozen=True)
class ErnComparisonResult:
    source: Path
    waveforms: dict[str, ErnWaveform]
    processing_spec: ProcessingSpec
    n_paired_trials: int
    n_epochs: int
    n_correct: int
    n_incorrect: int
    rejected_epochs: int
    ica_excluded_components: tuple[int, ...]
    wiener_diagnostics: dict
    baseline_ms: tuple[float, float]
    peak_window_ms: tuple[float, float]


def _default_erp_config_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "configs" / "erp_core_flankers.yaml"
    return Path(__file__).resolve().parents[2] / "configs" / "erp_core_flankers.yaml"


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


def build_response_trials(
    events: np.ndarray,
    event_id: dict[str, int],
    sfreq: float,
    max_lag_sec: float = 1.5,
) -> list[ResponseTrial]:
    """Pair responses with preceding Flankers stimuli like script 10."""
    descriptions = {value: key for key, value in event_id.items()}
    last_stimulus: tuple[int, str, str] | None = None
    trials: list[ResponseTrial] = []
    for sample, _, code in events:
        description = descriptions.get(int(code), "")
        stimulus = _stimulus_details(description)
        if stimulus is not None:
            last_stimulus = (int(sample), stimulus[0], stimulus[1])
            continue
        response = _response_details(description)
        if response is None or last_stimulus is None:
            continue
        stimulus_sample, compatibility, target_side = last_stimulus
        reaction_time = (int(sample) - stimulus_sample) / float(sfreq)
        if reaction_time < 0 or reaction_time > float(max_lag_sec):
            continue
        response_side, encoded_correct = response
        correct = target_side == response_side if encoded_correct is None else encoded_correct
        trials.append(
            ResponseTrial(
                sample=int(sample),
                correct=bool(correct),
                response_side=response_side,
                target_side=target_side,
                compatibility=compatibility,
                reaction_time_sec=float(reaction_time),
            )
        )
        last_stimulus = None
    if not trials:
        raise ValueError("没有找到有效的 Flankers 刺激—响应事件对")
    if len({trial.correct for trial in trials}) != 2:
        raise ValueError("ERN 分析同时需要正确和错误响应试次")
    return trials


def summarize_ern_epochs(epochs, correct: np.ndarray) -> ErnWaveform:
    correct = np.asarray(correct, dtype=bool)
    if len(correct) != len(epochs):
        raise ValueError("试次标签数量与 ERN epoch 数量不一致")
    if not np.any(correct) or not np.any(~correct):
        raise ValueError("共享伪迹剔除后缺少正确或错误响应试次")
    data = epochs.get_data(picks=["FCz"], copy=False)[:, 0]
    incorrect = data[~correct]
    correct_data = data[correct]
    incorrect_mean = incorrect.mean(axis=0)
    correct_mean = correct_data.mean(axis=0)
    return ErnWaveform(
        times_ms=epochs.times.copy() * 1000.0,
        incorrect_mean_uv=incorrect_mean * 1e6,
        difference_mean_uv=(incorrect_mean - correct_mean) * 1e6,
        incorrect_sd_uv=np.std(incorrect, axis=0) * 1e6,
        difference_sd_uv=(
            np.sqrt(np.var(incorrect, axis=0) + np.var(correct_data, axis=0)) * 1e6
        ),
    )


class ErnComparisonService:
    def __init__(
        self,
        recording_service: RecordingService | None = None,
        config_path: str | Path | None = None,
    ):
        self.recordings = recording_service or RecordingService()
        self.config_path = Path(config_path or _default_erp_config_path())

    @staticmethod
    def _check_cancel(cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelled("用户已取消 ERN 对比分析")

    def _build_config(self, spec: ProcessingSpec, n_channels: int) -> dict:
        spec.validate()
        cfg = load_config(self.config_path)
        cfg["preprocessing"]["target_sfreq"] = float(spec.target_sfreq)
        cfg["preprocessing"]["bandpass"] = [
            float(spec.bandpass_low_hz),
            float(spec.bandpass_high_hz),
        ]
        cfg["preprocessing"]["epoch_length_sec"] = float(spec.analysis_window_sec)
        cfg["wiener"]["mode"] = spec.wiener_mode.value
        cfg["wiener"]["coherence_threshold"] = float(spec.coherence_threshold)
        cfg["wiener"]["phase_gate_threshold_rad"] = spec.effective_phase_gate_rad
        cfg["wiener"]["freq_band"] = [
            float(spec.bandpass_low_hz),
            float(spec.bandpass_high_hz),
        ]
        cfg["wiener"]["nperseg"] = max(
            2,
            min(
                int(round(4.0 * spec.target_sfreq)),
                int(round(spec.analysis_window_sec * spec.target_sfreq)),
            ),
        )
        if spec.channel_groups is not None:
            cfg["channels"]["channel_groups"] = [
                list(group) for group in spec.channel_groups
            ]
        if spec.ica_n_components is not None:
            cfg["erp_core"]["standard_ica"]["n_components"] = min(
                int(spec.ica_n_components), max(1, n_channels - 1)
            )
        return cfg

    @staticmethod
    def _common_preprocess(raw, cfg: dict):
        common = raw.copy().pick("eeg")
        common.set_montage(
            "standard_1005", match_case=False, on_missing="ignore", verbose=False
        )
        line_freq = float(cfg["erp_core"]["line_freq"])
        if line_freq < common.info["sfreq"] / 2:
            common.notch_filter(line_freq, verbose=False)
        low, high = map(float, cfg["preprocessing"]["bandpass"])
        common.filter(low, high, method="fir", verbose=False)
        target_sfreq = float(cfg["preprocessing"]["target_sfreq"])
        if not np.isclose(common.info["sfreq"], target_sfreq):
            common.resample(target_sfreq, verbose=False)
        return common

    @staticmethod
    def _apply_standard_ica(raw, cfg: dict):
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
        proxies = [
            channel
            for channel in ica_cfg["eog_proxy_channels"]
            if channel in raw.ch_names
        ]
        excluded: list[int] = []
        for proxy in proxies:
            indices, _ = ica.find_bads_eog(fit_raw, ch_name=proxy, verbose=False)
            excluded.extend(indices)
        ica.exclude = sorted(set(excluded))
        return (
            ica.apply(raw.copy(), verbose=False),
            tuple(int(value) for value in ica.exclude),
        )

    @staticmethod
    def _make_shared_epochs(raws: dict, trials: list[ResponseTrial], cfg: dict):
        spec = cfg["erp_core"]["ern"]
        events = np.column_stack(
            [
                np.array([trial.sample for trial in trials], dtype=int),
                np.zeros(len(trials), dtype=int),
                np.ones(len(trials), dtype=int),
            ]
        )
        probe = mne.Epochs(
            raws["raw"],
            events,
            event_id={"response": 1},
            tmin=float(spec["tmin"]),
            tmax=float(spec["tmax"]),
            baseline=None,
            reject={"eeg": float(cfg["preprocessing"]["artifact_threshold_uv"]) * 1e-6},
            preload=True,
            verbose=False,
        )
        selected_events = events[probe.selection]
        selected_correct = np.array(
            [trials[index].correct for index in probe.selection], dtype=bool
        )
        baseline = tuple(float(value) for value in spec["baseline"])
        epochs = {
            method: mne.Epochs(
                branch,
                selected_events,
                event_id={"response": 1},
                tmin=float(spec["tmin"]),
                tmax=float(spec["tmax"]),
                baseline=baseline,
                preload=True,
                reject_by_annotation=False,
                verbose=False,
            )
            for method, branch in raws.items()
        }
        return epochs, selected_correct

    def compare(
        self,
        source: str | Path,
        processing: ProcessingSpec,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        progress: ProgressCallback | None = None,
    ) -> ErnComparisonResult:
        source = Path(source).resolve()
        raw, _ = self.recordings.load_eeg(source, preload=True)
        if "FCz" not in raw.ch_names:
            raise ValueError("ERN 三方法叠加需要 FCz 通道")
        if processing.bandpass_high_hz >= float(raw.info["sfreq"]) / 2:
            raise ValueError("带通高限必须低于源文件采样率的 Nyquist 频率")
        cfg = self._build_config(processing, len(raw.ch_names))
        self._check_cancel(cancel_requested)
        if progress is not None:
            progress(1, 4, "共享滤波与重采样")
        common = self._common_preprocess(raw, cfg)
        events, event_id = mne.events_from_annotations(common, verbose=False)
        trials = build_response_trials(
            events,
            event_id,
            common.info["sfreq"],
            float(cfg["erp_core"]["response_pairing_window_sec"]),
        )

        self._check_cancel(cancel_requested)
        if progress is not None:
            progress(2, 4, "标准 ICA 全记录拟合")
        ica, excluded = self._apply_standard_ica(common, cfg)

        self._check_cancel(cancel_requested)
        subject_id = source.stem
        wiener, diagnostics = wiener_continuous_raw(
            common,
            cfg,
            subject_id,
            cancel_requested=cancel_requested,
            progress=(
                (lambda current, total: progress(current, total, "ECMAD 窗口"))
                if progress is not None
                else None
            ),
        )

        self._check_cancel(cancel_requested)
        if progress is not None:
            progress(4, 4, "响应锁定 ERN")
        epochs, correct = self._make_shared_epochs(
            {"raw": common, "ica": ica, "wiener": wiener}, trials, cfg
        )
        if not np.any(correct) or not np.any(~correct):
            raise ValueError("共享伪迹剔除后缺少正确或错误响应试次")
        waveforms = {
            method: summarize_ern_epochs(method_epochs, correct)
            for method, method_epochs in epochs.items()
        }
        ern_cfg = cfg["erp_core"]["ern"]
        return ErnComparisonResult(
            source=source,
            waveforms=waveforms,
            processing_spec=processing,
            n_paired_trials=len(trials),
            n_epochs=len(correct),
            n_correct=int(np.count_nonzero(correct)),
            n_incorrect=int(np.count_nonzero(~correct)),
            rejected_epochs=len(trials) - len(correct),
            ica_excluded_components=excluded,
            wiener_diagnostics=diagnostics,
            baseline_ms=tuple(float(value) * 1000.0 for value in ern_cfg["baseline"]),
            peak_window_ms=tuple(
                float(value) * 1000.0 for value in ern_cfg["peak_window"]
            ),
        )
