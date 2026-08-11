from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eeg_bg.exceptions import ProcessingCancelled


class ExtractionMode(str, Enum):
    SELECTION = "selection"
    FIXED_WINDOWS = "fixed_windows"
    CONTINUOUS = "continuous"


class ProcessingMethod(str, Enum):
    BASIC = "basic"
    ICA = "ica"
    WIENER = "wiener"


class WienerMode(str, Enum):
    FREQUENCY = "frequency"
    PHASEGATED = "phasegated"
    ZEROPHASE = "zerophase"


class OutputFormat(str, Enum):
    EDF = "edf"
    FIF = "fif"


@dataclass(frozen=True)
class ArtifactSettings:
    enabled: bool = True
    threshold_uv: float = 200.0

    def validate(self) -> None:
        if not math.isfinite(self.threshold_uv) or self.threshold_uv <= 0:
            raise ValueError("伪迹阈值必须为正数")

    def as_serializable_dict(self) -> dict[str, Any]:
        return {"enabled": bool(self.enabled), "threshold_uv": float(self.threshold_uv)}


@dataclass(frozen=True)
class ExtractionSpec:
    mode: ExtractionMode = ExtractionMode.SELECTION
    start_sec: float = 0.0
    stop_sec: float | None = 20.0
    window_sec: float = 20.0

    def __post_init__(self) -> None:
        # Qt stores str-backed enums as plain strings in QVariant. Normalize at
        # the application boundary so serialization and processing stay typed.
        object.__setattr__(self, "mode", ExtractionMode(self.mode))

    def validate(self, duration_sec: float | None = None) -> None:
        if self.window_sec < 4.0:
            raise ValueError("分析窗长必须至少为 4 秒")
        if self.mode == ExtractionMode.SELECTION:
            if self.stop_sec is None or self.start_sec < 0 or self.stop_sec <= self.start_sec:
                raise ValueError("交互选区必须满足 0 ≤ 起点 < 终点")
            if duration_sec is not None and self.stop_sec > duration_sec + 1e-9:
                raise ValueError("选区终点超出记录时长")

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True)
class ProcessingSpec:
    method: ProcessingMethod = ProcessingMethod.BASIC
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    target_sfreq: float = 125.0
    analysis_window_sec: float = 20.0
    ica_n_components: int | None = None
    ica_artifact_corr_threshold: float = 0.80
    wiener_mode: WienerMode = WienerMode.FREQUENCY
    coherence_threshold: float = 0.15
    coherent_gate_enabled: bool = True
    coherent_gate_threshold_uv: float = 100.0
    phase_gate_threshold_rad: float = 0.39
    protected_band_hz: tuple[float, float] | None = (5.0, 20.0)
    channel_groups: tuple[tuple[str, ...], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", ProcessingMethod(self.method))
        object.__setattr__(self, "wiener_mode", WienerMode(self.wiener_mode))
        if self.protected_band_hz is not None:
            object.__setattr__(
                self,
                "protected_band_hz",
                tuple(float(value) for value in self.protected_band_hz),
            )
        if self.channel_groups is not None:
            object.__setattr__(
                self,
                "channel_groups",
                tuple(
                    tuple(str(channel) for channel in group)
                    for group in self.channel_groups
                ),
            )

    def validate(self) -> None:
        values = (
            self.bandpass_low_hz,
            self.bandpass_high_hz,
            self.target_sfreq,
            self.analysis_window_sec,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("基础预处理参数必须为有限数值")
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < self.target_sfreq / 2:
            raise ValueError("带通范围必须满足 0 < low < high < 目标采样率 / 2")
        if self.analysis_window_sec < 4.0:
            raise ValueError("分析窗长必须至少为 4 秒")
        if self.ica_n_components is not None and self.ica_n_components < 1:
            raise ValueError("ICA 组件数必须为正整数或自动")
        if not 0.0 <= self.ica_artifact_corr_threshold <= 1.0:
            raise ValueError("ICA 相关阈值必须位于 [0, 1]")
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError("coherence 必须位于 [0, 1]")
        if not isinstance(self.coherent_gate_enabled, bool):
            raise ValueError("coherent 功率门控开关必须为布尔值")
        if (
            isinstance(self.coherent_gate_threshold_uv, bool)
            or not math.isfinite(float(self.coherent_gate_threshold_uv))
            or float(self.coherent_gate_threshold_uv) <= 0.0
        ):
            raise ValueError("coherent 功率门控阈值必须为有限正数")
        if not math.isfinite(self.phase_gate_threshold_rad) or not 0.0 <= self.phase_gate_threshold_rad <= 3.14:
            raise ValueError("gate 必须位于 [0, π]")
        if (
            self.method == ProcessingMethod.WIENER
            and self.protected_band_hz is not None
        ):
            if len(self.protected_band_hz) != 2:
                raise ValueError("保护频带必须包含低限和高限")
            low_hz, high_hz = self.protected_band_hz
            if not math.isfinite(low_hz) or not math.isfinite(high_hz):
                raise ValueError("保护频带必须为有限数值")
            if not (
                self.bandpass_low_hz
                <= low_hz
                < high_hz
                <= self.bandpass_high_hz
            ):
                raise ValueError("保护频带必须位于带通范围内且满足 low < high")
            if (
                math.isclose(low_hz, self.bandpass_low_hz)
                and math.isclose(high_hz, self.bandpass_high_hz)
            ):
                raise ValueError("保护频带不能覆盖全部带通范围")
        if self.method == ProcessingMethod.WIENER and self.channel_groups is not None:
            if not self.channel_groups:
                raise ValueError("ECMAD 至少需要一个导联组")
            for index, group in enumerate(self.channel_groups, start=1):
                if len(group) < 2:
                    raise ValueError(f"导联组 G{index} 至少需要两个导联")
                if any(not channel.strip() for channel in group):
                    raise ValueError(f"导联组 G{index} 包含空导联名")
                if len(set(group)) != len(group):
                    raise ValueError(f"导联组 G{index} 不能包含重复导联")

    @property
    def effective_phase_gate_rad(self) -> float:
        # The two-decimal GUI endpoint represents the exact all-pass boundary.
        if math.isclose(self.phase_gate_threshold_rad, 3.14, abs_tol=5e-3):
            return math.pi
        return float(self.phase_gate_threshold_rad)

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["method"] = self.method.value
        payload["wiener_mode"] = self.wiener_mode.value
        payload["phase_gate_threshold_rad"] = self.effective_phase_gate_rad
        if self.protected_band_hz is not None:
            payload["protected_band_hz"] = list(self.protected_band_hz)
        if self.channel_groups is not None:
            payload["channel_groups"] = [list(group) for group in self.channel_groups]
        return payload

    def as_fingerprint_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method.value,
            "bandpass_low_hz": self.bandpass_low_hz,
            "bandpass_high_hz": self.bandpass_high_hz,
            "target_sfreq": self.target_sfreq,
            "analysis_window_sec": self.analysis_window_sec,
        }
        if self.method == ProcessingMethod.ICA:
            payload.update({
                "ica_n_components": self.ica_n_components,
                "ica_artifact_corr_threshold": self.ica_artifact_corr_threshold,
            })
        elif self.method == ProcessingMethod.WIENER:
            payload.update({
                "wiener_mode": self.wiener_mode.value,
                "coherence_threshold": self.coherence_threshold,
                "coherent_gate_enabled": self.coherent_gate_enabled,
                "coherent_gate_threshold_uv": self.coherent_gate_threshold_uv,
                "protected_band_hz": self.protected_band_hz,
                "channel_groups": self.channel_groups,
            })
            if self.wiener_mode != WienerMode.FREQUENCY:
                payload["phase_gate_threshold_rad"] = self.effective_phase_gate_rad
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_fingerprint_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def pipeline_fingerprint(processing: ProcessingSpec, extraction: ExtractionSpec) -> str:
    encoded = json.dumps(
        {
            "processing": processing.as_fingerprint_dict(),
            "extraction": extraction.as_serializable_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class RecordingEvent:
    onset_sec: float
    duration_sec: float = 0.0
    sample: int | None = None
    trial_type: str = ""
    value: str = ""
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class RecordingSidecars:
    eeg: dict[str, Any] = field(default_factory=dict)
    channels: list[dict[str, str]] = field(default_factory=list)
    events: list[RecordingEvent] = field(default_factory=list)
    electrodes: list[dict[str, str]] = field(default_factory=list)
    coordsystem: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class RecordingInfo:
    path: Path
    format: str
    ch_names: list[str]
    sfreq: float
    duration_sec: float
    n_times: int
    warnings: list[str] = field(default_factory=list)
    sidecars: RecordingSidecars = field(default_factory=RecordingSidecars)


@dataclass
class ProcessedSegment:
    raw: Any
    start_sec: float
    stop_sec: float
    window_index: int | None = None


@dataclass
class ProcessingResult:
    source: Path
    original_raw: Any
    processed_segments: list[ProcessedSegment]
    info: RecordingInfo
    processing_spec: ProcessingSpec
    extraction_spec: ExtractionSpec
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def preview_raw(self) -> Any:
        if not self.processed_segments:
            raise ValueError("处理结果不包含可预览片段")
        return self.processed_segments[0].raw


@dataclass
class BatchItemResult:
    source: Path
    status: str
    outputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_sec: float = 0.0
    config_hash: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
