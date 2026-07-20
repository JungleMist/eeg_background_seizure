from __future__ import annotations

from pathlib import Path
import os
import uuid

import mne
import numpy as np

from .models import OutputFormat, RecordingInfo


SUPPORTED_SUFFIXES = (".edf", ".fif", ".fif.gz")


def _normalize_channel_name(raw_name: str) -> str:
    name = raw_name.strip()
    if name.upper().startswith("EEG "):
        name = name[4:].strip()
    if "-" in name:
        name = name.split("-", 1)[0].strip()
    return name.upper()


def recording_format(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith(".edf"):
        return "edf"
    if name.endswith(".fif") or name.endswith(".fif.gz"):
        return "fif"
    raise ValueError(f"不支持的 EEG 文件格式：{Path(path).name}")


def is_supported_recording(path: str | Path) -> bool:
    try:
        recording_format(path)
    except ValueError:
        return False
    return True


class RecordingService:
    def __init__(self, standard_channels: list[str] | None = None):
        self.standard_channels = standard_channels or [
            "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
            "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
            "Fz", "Cz", "Pz",
        ]

    def open_raw(self, path: str | Path, *, preload: bool = False):
        path = Path(path)
        fmt = recording_format(path)
        if not path.is_file():
            raise FileNotFoundError(f"EEG 文件不存在：{path}")
        if fmt == "edf":
            return mne.io.read_raw_edf(str(path), preload=preload, verbose=False)
        return mne.io.read_raw_fif(str(path), preload=preload, verbose=False)

    def _prepare_eeg_channels(self, raw) -> list[str]:
        warnings: list[str] = []
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise ValueError("文件中没有标记为 EEG 的通道")
        raw.pick(eeg_picks)

        canonical = {name.upper(): name for name in self.standard_channels}
        rename: dict[str, str] = {}
        occupied = set(raw.ch_names)
        for original in raw.ch_names:
            normalized = _normalize_channel_name(original)
            target = canonical.get(normalized)
            if target is None or target == original:
                continue
            if target in occupied and target != original:
                warnings.append(f"通道 {original} 不能规范化为 {target}：名称冲突")
                continue
            rename[original] = target
            occupied.add(target)
        if rename:
            raw.rename_channels(rename)
        return warnings

    def inspect(self, path: str | Path) -> RecordingInfo:
        raw = self.open_raw(path, preload=False)
        warnings = self._prepare_eeg_channels(raw)
        return RecordingInfo(
            path=Path(path).resolve(),
            format=recording_format(path),
            ch_names=list(raw.ch_names),
            sfreq=float(raw.info["sfreq"]),
            duration_sec=float(raw.n_times / raw.info["sfreq"]),
            n_times=int(raw.n_times),
            warnings=warnings,
        )

    def load_eeg(self, path: str | Path, *, preload: bool = True):
        raw = self.open_raw(path, preload=preload)
        warnings = self._prepare_eeg_channels(raw)
        return raw, warnings

    def apply_basic_preprocessing(self, raw, low: float, high: float, sfreq: float):
        processed = raw.copy().load_data()
        processed.filter(
            low,
            high,
            method="iir",
            iir_params=dict(order=5, ftype="butter"),
            verbose=False,
        )
        if not np.isclose(processed.info["sfreq"], sfreq):
            processed.resample(float(sfreq), verbose=False)
        return processed

    def write(self, raw, out_path: str | Path, output_format: OutputFormat) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:10]
        if output_format == OutputFormat.EDF:
            tmp_path = out_path.with_name(f".{out_path.stem}.{token}.tmp.edf")
            try:
                raw.export(str(tmp_path), fmt="edf", overwrite=True, verbose=False)
                os.replace(tmp_path, out_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            return

        tmp_path = out_path.with_name(f".{out_path.stem}.{token}-raw.fif")
        try:
            raw.save(str(tmp_path), overwrite=True, verbose=False)
            os.replace(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
