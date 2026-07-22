from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.ica import apply_ica, fit_ica
from eeg_bg.decomposition.wiener import CANDIDATE_SOLVE_FAILED
from eeg_bg.preprocessing.continuous import (
    candidate_diagnostics,
    select_wiener_decomposer,
    wiener_continuous_raw,
)

from .models import (
    ExtractionMode,
    ExtractionSpec,
    ProcessedSegment,
    ProcessingCancelled,
    ProcessingMethod,
    ProcessingResult,
    ProcessingSpec,
)
from .recording import RecordingService


ProgressCallback = Callable[[int, int, str], None]


def _default_config_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "configs" / "default.yaml"
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ProcessingEngine:
    def __init__(
        self,
        recording_service: RecordingService | None = None,
        base_config_path: str | Path | None = None,
    ):
        self.base_config_path = Path(base_config_path or _default_config_path())
        self.base_cfg = load_config(self.base_config_path)
        self.recordings = recording_service or RecordingService(
            list(self.base_cfg["channels"]["standard_19"])
        )

    def build_config(self, spec: ProcessingSpec, n_channels: int) -> dict:
        spec.validate()
        cfg = deepcopy(self.base_cfg)
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
        auto_components = max(1, min(19, n_channels - 1))
        cfg["ica"]["n_components"] = min(
            int(spec.ica_n_components or auto_components), auto_components
        )
        cfg["ica"]["artifact_corr_threshold"] = float(
            spec.ica_artifact_corr_threshold
        )
        return cfg

    def _check_cancel(self, cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelled("用户已取消处理")

    @staticmethod
    def _crop(raw, start_sec: float, stop_sec: float):
        # include_tmax=False avoids duplicating one sample at adjacent windows.
        return raw.copy().crop(
            tmin=float(start_sec),
            tmax=float(stop_sec),
            include_tmax=False,
        )

    @staticmethod
    def _window_epochs(raw, window_sec: float):
        sfreq = float(raw.info["sfreq"])
        n_times = int(round(window_sec * sfreq))
        data_uv = raw.get_data() * 1e6
        windows: list[tuple[int, np.ndarray]] = []
        for idx, start in enumerate(range(0, data_uv.shape[1] - n_times + 1, n_times)):
            epoch = data_uv[:, start : start + n_times]
            windows.append((idx, epoch.copy()))
        return windows, n_times

    def _fit_ica(
        self,
        raw,
        cfg: dict,
        fit_window_sec: float,
    ):
        epochs, _ = self._window_epochs(raw, fit_window_sec)
        if not epochs:
            # A valid interactive selection may be shorter than the fixed export
            # window. It is still suitable as one ICA fitting block.
            data_uv = raw.get_data() * 1e6
            if data_uv.size:
                epochs = [(0, data_uv)]
        if not epochs:
            raise ValueError("没有可用于 ICA 拟合的数据")
        batch = np.stack([epoch for _, epoch in epochs])
        model, artifact_indices = fit_ica(batch, list(raw.ch_names), cfg)
        return model, artifact_indices, len(epochs)

    def _apply_method_to_raw(
        self,
        raw,
        cfg: dict,
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        *,
        cancel_requested: Callable[[], bool] | None,
        progress: ProgressCallback | None,
        subject_id: str,
        ica_fit_raw=None,
    ):
        if processing.method == ProcessingMethod.BASIC:
            return raw.copy(), {"method": "basic"}, []

        if processing.method == ProcessingMethod.ICA:
            if len(raw.ch_names) < 2:
                raise ValueError("ICA 至少需要 2 个 EEG 通道")
            warnings: list[str] = []
            if not any(ch in raw.ch_names for ch in ("FP1", "FP2")):
                warnings.append("缺少 FP1/FP2：ICA 未找到自动伪迹代理，输出可能与基础处理相同")
            model, artifacts, fit_windows = self._fit_ica(
                ica_fit_raw if ica_fit_raw is not None else raw,
                cfg,
                processing.analysis_window_sec,
            )
            self._check_cancel(cancel_requested)
            data_uv = raw.get_data()[None, ...] * 1e6
            cleaned_uv = apply_ica(data_uv, model, artifacts, list(raw.ch_names), cfg)[0]
            cleaned = raw.copy()
            cleaned._data = cleaned_uv * 1e-6
            return cleaned, {
                "method": "ica",
                "removed_components": [int(i) for i in artifacts],
                "fit_windows": int(fit_windows),
            }, warnings

        complete_groups = [
            group for group in cfg["channels"]["channel_groups"]
            if all(ch in raw.ch_names for ch in group)
        ]
        if not complete_groups:
            raise ValueError("当前通道中没有可用于 Wiener 的完整传导通道组")
        missing_groups = [
            "-".join(group) for group in cfg["channels"]["channel_groups"]
            if group not in complete_groups
        ]
        warnings = []
        if missing_groups:
            warnings.append("缺失的 Wiener 通道组将保持原样：" + ", ".join(missing_groups))
        denoised, diagnostics = wiener_continuous_raw(
            raw,
            cfg,
            subject_id,
            cancel_requested=cancel_requested,
            progress=(
                (lambda current, total: progress(current, total, "Wiener 窗口"))
                if progress is not None else None
            ),
        )
        if int(diagnostics.get("solve_failures", 0)):
            warnings.append(
                f"{diagnostics['solve_failures']} 个 Wiener 候选求解失败；对应通道已安全直通"
            )
        return denoised, diagnostics, warnings

    def _process_fixed_windows(
        self,
        raw,
        cfg: dict,
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        *,
        cancel_requested: Callable[[], bool] | None,
        progress: ProgressCallback | None,
        subject_id: str,
    ):
        windows, n_times = self._window_epochs(raw, extraction.window_sec)
        if not windows:
            raise ValueError("记录时长不足一个完整的固定窗口")

        ica_state = None
        warnings: list[str] = []
        if processing.method == ProcessingMethod.ICA:
            if not any(ch in raw.ch_names for ch in ("FP1", "FP2")):
                warnings.append("缺少 FP1/FP2：ICA 未找到自动伪迹代理")
            batch = np.stack([epoch for _, epoch in windows])
            ica_state = fit_ica(batch, list(raw.ch_names), cfg)

        decomposer = (
            select_wiener_decomposer(processing.wiener_mode.value)
            if processing.method == ProcessingMethod.WIENER else None
        )
        if decomposer is not None:
            complete_groups = [
                group for group in cfg["channels"]["channel_groups"]
                if all(ch in raw.ch_names for ch in group)
            ]
            if not complete_groups:
                raise ValueError("当前通道中没有可用于 Wiener 的完整传导通道组")
            missing_groups = [
                "-".join(group) for group in cfg["channels"]["channel_groups"]
                if group not in complete_groups
            ]
            if missing_groups:
                warnings.append(
                    "缺失的 Wiener 通道组将保持原样：" + ", ".join(missing_groups)
                )

        segments: list[ProcessedSegment] = []
        solve_failures = 0
        window_diagnostics: list[dict] = []
        sfreq = float(raw.info["sfreq"])
        for pos, (window_index, epoch_uv) in enumerate(windows):
            self._check_cancel(cancel_requested)
            start = window_index * extraction.window_sec
            processed_uv = epoch_uv
            if ica_state is not None:
                model, artifacts = ica_state
                processed_uv = apply_ica(
                    epoch_uv[None, ...], model, artifacts, list(raw.ch_names), cfg
                )[0]
            elif decomposer is not None:
                result = decomposer(
                    epoch_uv,
                    list(raw.ch_names),
                    cfg,
                    subject_id=subject_id,
                    epoch_idx=window_index,
                )
                processed_uv = result.specific
                if result.candidate_status is not None:
                    solve_failures += int(np.count_nonzero(
                        result.candidate_status == CANDIDATE_SOLVE_FAILED
                    ))
                window_diagnostics.append({
                    "window_index": window_index,
                    "start_sample": int(round(start * sfreq)),
                    **candidate_diagnostics(result),
                })
            segment_raw = self._crop(raw, start, start + extraction.window_sec)
            segment_raw._data = processed_uv * 1e-6
            segments.append(
                ProcessedSegment(
                    raw=segment_raw,
                    start_sec=start,
                    stop_sec=start + n_times / sfreq,
                    window_index=window_index,
                )
            )
            if progress is not None:
                progress(pos + 1, len(windows), "固定窗口")
        if solve_failures:
            warnings.append(f"{solve_failures} 个 Wiener 候选求解失败；对应通道已安全直通")
        return segments, {
            "method": processing.method.value,
            "processed_windows": len(windows),
            "incomplete_tail_samples": int(raw.n_times - len(windows) * n_times),
            "solve_failures": solve_failures,
            "window_diagnostics": window_diagnostics,
        }, warnings

    def process(
        self,
        source: str | Path,
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        processing.validate()
        info = self.recordings.inspect(source)
        extraction.validate(info.duration_sec)
        self._check_cancel(cancel_requested)
        original_raw, load_warnings = self.recordings.load_eeg(source, preload=True)
        if processing.bandpass_high_hz >= float(original_raw.info["sfreq"]) / 2:
            raise ValueError("带通高限必须低于源文件采样率的 Nyquist 频率")
        cfg = self.build_config(processing, len(original_raw.ch_names))
        processed_base = self.recordings.apply_basic_preprocessing(
            original_raw,
            processing.bandpass_low_hz,
            processing.bandpass_high_hz,
            processing.target_sfreq,
        )
        self._check_cancel(cancel_requested)
        subject_id = Path(source).stem

        warnings = list(dict.fromkeys(info.warnings + load_warnings))
        if extraction.mode == ExtractionMode.FIXED_WINDOWS:
            segments, diagnostics, method_warnings = self._process_fixed_windows(
                processed_base,
                cfg,
                processing,
                extraction,
                cancel_requested=cancel_requested,
                progress=progress,
                subject_id=subject_id,
            )
        else:
            target = processed_base
            start_sec = 0.0
            stop_sec = float(processed_base.n_times / processed_base.info["sfreq"])
            if extraction.mode == ExtractionMode.SELECTION:
                start_sec = float(extraction.start_sec)
                stop_sec = float(extraction.stop_sec)
                target = self._crop(processed_base, start_sec, stop_sec)
            processed, diagnostics, method_warnings = self._apply_method_to_raw(
                target,
                cfg,
                processing,
                extraction,
                cancel_requested=cancel_requested,
                progress=progress,
                subject_id=subject_id,
                ica_fit_raw=processed_base,
            )
            segments = [
                ProcessedSegment(
                    raw=processed,
                    start_sec=start_sec,
                    stop_sec=stop_sec,
                )
            ]
        warnings.extend(method_warnings)
        return ProcessingResult(
            source=Path(source).resolve(),
            original_raw=original_raw,
            processed_segments=segments,
            info=info,
            processing_spec=processing,
            extraction_spec=extraction,
            warnings=list(dict.fromkeys(warnings)),
            diagnostics=diagnostics,
        )
