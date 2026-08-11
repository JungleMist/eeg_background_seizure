from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable
import uuid

from .models import (
    ArtifactSettings,
    BatchItemResult,
    ExtractionSpec,
    OutputFormat,
    ProcessingCancelled,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
    pipeline_fingerprint,
)
from .artifacts import summarize_raw_artifacts
from .processing import ProcessingEngine
from .recording import is_supported_recording


LOGGER = logging.getLogger(__name__)


def scan_recordings(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{root}")
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink())
        for name in sorted(files):
            path = Path(current) / name
            if is_supported_recording(path):
                found.append(path)
    return found


def validate_batch_roots(input_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    source = Path(input_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("输出目录不能与输入目录相同，也不能位于输入目录内部")
    return source, output


def _source_stem(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".fif.gz"):
        return path.name[:-7]
    return path.stem


def output_label(spec: ProcessingSpec) -> str:
    if spec.method == ProcessingMethod.WIENER:
        label = f"wiener-{spec.wiener_mode.value}__c{spec.coherence_threshold:.2f}"
        label += (
            f"_cg{spec.coherent_gate_threshold_uv:g}"
            if spec.coherent_gate_enabled
            else "_cg-off"
        )
        if spec.wiener_mode != WienerMode.FREQUENCY:
            label += f"_g{spec.phase_gate_threshold_rad:.2f}"
        if spec.protected_band_hz is None:
            label += "_pb-off"
        else:
            low_hz, high_hz = spec.protected_band_hz
            label += f"_pb{low_hz:g}-{high_hz:g}"
        return label
    return spec.method.value


def output_name(
    source: Path,
    spec: ProcessingSpec,
    output_format: OutputFormat,
    *,
    extraction: ExtractionSpec | None = None,
    window_index: int | None = None,
) -> str:
    suffix = ".edf" if output_format == OutputFormat.EDF else ".fif"
    window = f"__w{window_index + 1:05d}" if window_index is not None else ""
    fingerprint = (
        pipeline_fingerprint(spec, extraction)
        if extraction is not None
        else spec.fingerprint
    )
    return (
        f"{_source_stem(source)}__{output_label(spec)}"
        f"__p{fingerprint[:5]}{window}{suffix}"
    )


def _jsonable_extraction(spec: ExtractionSpec) -> dict:
    return spec.as_serializable_dict()


class BatchProcessor:
    def __init__(self, engine: ProcessingEngine | None = None):
        self.engine = engine or ProcessingEngine()

    def run(
        self,
        files: list[Path],
        input_root: str | Path,
        output_root: str | Path,
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        output_format: OutputFormat,
        *,
        artifact_settings: ArtifactSettings | None = None,
        overwrite: bool = False,
        cancel_requested: Callable[[], bool] | None = None,
        item_progress: Callable[[int, int, Path], None] | None = None,
        stage_progress: Callable[[int, int, str], None] | None = None,
        item_finished: Callable[[BatchItemResult], None] | None = None,
    ) -> list[BatchItemResult]:
        input_root, output_root = validate_batch_roots(input_root, output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_settings = artifact_settings or ArtifactSettings()
        artifact_settings.validate()
        results: list[BatchItemResult] = []
        config_hash = pipeline_fingerprint(processing, extraction)
        for index, source in enumerate(files):
            source = Path(source).expanduser().resolve()
            if cancel_requested is not None and cancel_requested():
                break
            if item_progress is not None:
                item_progress(index, len(files), source)
            started = time.perf_counter()
            item = BatchItemResult(
                source=source,
                status="running",
                config_hash=config_hash,
            )
            cancelled = False
            try:
                result = self.engine.process(
                    source,
                    processing,
                    extraction,
                    cancel_requested=cancel_requested,
                    progress=stage_progress,
                )
                artifact_summary = summarize_raw_artifacts(
                    result.original_raw, artifact_settings
                )
                if not artifact_settings.enabled:
                    artifact_summary.update({
                        "affected_channels": [],
                        "affected_channel_count": 0,
                        "exceedance_region_count": 0,
                        "channel_region_counts": {},
                    })
                artifact_warnings: list[str] = []
                if artifact_settings.enabled and artifact_summary["affected_channels"]:
                    warning = (
                        f"原始输入超过 {artifact_settings.threshold_uv:g} µV："
                        f"通道 {', '.join(artifact_summary['affected_channels'])}，"
                        f"{artifact_summary['exceedance_region_count']} 个连续区段，"
                        f"最大绝对振幅 {artifact_summary['max_abs_uv']:.1f} µV"
                    )
                    artifact_warnings.append(warning)
                    LOGGER.warning("%s：%s", source, warning)
                relative_parent = source.parent.relative_to(input_root)
                destination = output_root / relative_parent
                outputs: list[Path] = []
                skipped = 0
                for segment in result.processed_segments:
                    target = destination / output_name(
                        source,
                        processing,
                        output_format,
                        extraction=extraction,
                        window_index=segment.window_index,
                    )
                    if target.exists() and not overwrite:
                        outputs.append(target)
                        skipped += 1
                        continue
                    self.engine.recordings.write(segment.raw, target, output_format)
                    outputs.append(target)
                item.outputs = outputs
                item.warnings = list(result.warnings) + artifact_warnings
                item.diagnostics = {
                    **result.diagnostics,
                    "artifact_threshold": artifact_summary,
                }
                if skipped == len(outputs):
                    item.status = "skipped"
                elif item.warnings:
                    item.status = "warning"
                else:
                    item.status = "done"
            except ProcessingCancelled as exc:
                item.status = "cancelled"
                item.error = str(exc)
                cancelled = True
            except Exception as exc:
                item.status = "failed"
                item.error = str(exc)
            finally:
                item.elapsed_sec = time.perf_counter() - started
            if not results or results[-1] is not item:
                results.append(item)
            if item_finished is not None:
                item_finished(item)
            if cancelled:
                break
        self.write_manifest(
            output_root,
            results,
            processing,
            extraction,
            output_format,
            artifact_settings,
        )
        return results

    @staticmethod
    def write_manifest(
        output_root: Path,
        results: list[BatchItemResult],
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        output_format: OutputFormat,
        artifact_settings: ArtifactSettings | None = None,
    ) -> tuple[Path, Path]:
        artifact_settings = artifact_settings or ArtifactSettings()
        config_hash = pipeline_fingerprint(processing, extraction)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        stem = f"eeg_bg_manifest_{timestamp}_{config_hash[:8]}"
        json_path = output_root / f"{stem}.json"
        csv_path = output_root / f"{stem}.csv"
        rows = [
            {
                "source": str(item.source),
                "status": item.status,
                "outputs": [str(path) for path in item.outputs],
                "warnings": item.warnings,
                "error": item.error,
                "elapsed_sec": item.elapsed_sec,
                "config_hash": item.config_hash,
                "diagnostics": item.diagnostics,
            }
            for item in results
        ]
        payload = {
            "processing": processing.as_serializable_dict(),
            "extraction": _jsonable_extraction(extraction),
            "artifact_settings": artifact_settings.as_serializable_dict(),
            "output_format": output_format.value,
            "items": rows,
        }
        token = uuid.uuid4().hex[:8]
        tmp_json = json_path.with_name(f".{json_path.name}.{token}.tmp")
        tmp_csv = csv_path.with_name(f".{csv_path.name}.{token}.tmp")
        try:
            tmp_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with tmp_csv.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source", "status", "outputs", "warnings", "error",
                        "elapsed_sec", "config_hash", "diagnostics",
                        "artifact_threshold_enabled", "artifact_threshold_uv",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        **row,
                        "outputs": " | ".join(row["outputs"]),
                        "warnings": " | ".join(row["warnings"]),
                        "diagnostics": json.dumps(
                            row["diagnostics"], ensure_ascii=False, separators=(",", ":")
                        ),
                        "artifact_threshold_enabled": artifact_settings.enabled,
                        "artifact_threshold_uv": artifact_settings.threshold_uv,
                    })
            os.replace(tmp_json, json_path)
            os.replace(tmp_csv, csv_path)
        finally:
            tmp_json.unlink(missing_ok=True)
            tmp_csv.unlink(missing_ok=True)
        return json_path, csv_path
