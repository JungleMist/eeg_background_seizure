import json
from pathlib import Path

import mne

from eeg_bg.application.batch import (
    BatchProcessor,
    output_label,
    scan_recordings,
    validate_batch_roots,
)
from eeg_bg.application.models import (
    ArtifactSettings,
    ExtractionMode,
    ExtractionSpec,
    OutputFormat,
    ProcessingMethod,
    ProcessingSpec,
)


def test_scan_recordings_is_recursive_and_format_limited(synthetic_fif, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    copied = nested / synthetic_fif.name
    copied.write_bytes(synthetic_fif.read_bytes())
    eeglab_set = nested / "subject.set"
    eeglab_set.write_bytes(b"set header")
    (nested / "subject.fdt").write_bytes(b"sample data")
    (nested / "ignore.csv").write_text("x", encoding="utf-8")
    found = scan_recordings(tmp_path)
    assert synthetic_fif in found
    assert copied in found
    assert eeglab_set in found
    assert all(path.suffix in {".fif", ".edf", ".set"} for path in found)
    assert not any(path.suffix == ".fdt" for path in found)


def test_wiener_output_label_includes_coherent_gate():
    enabled = output_label(ProcessingSpec(
        method=ProcessingMethod.WIENER,
        coherent_gate_threshold_uv=250.0,
    ))
    disabled = output_label(ProcessingSpec(
        method=ProcessingMethod.WIENER,
        coherent_gate_enabled=False,
    ))

    assert "_cg250" in enabled
    assert "_cg-off" in disabled


def test_batch_mirrors_tree_and_writes_manifest(synthetic_fif, tmp_path):
    input_root = tmp_path / "input"
    nested = input_root / "subject"
    nested.mkdir(parents=True)
    source = nested / synthetic_fif.name
    source.write_bytes(synthetic_fif.read_bytes())
    output_root = tmp_path / "output"
    results = BatchProcessor().run(
        [source],
        input_root,
        output_root,
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(
            mode=ExtractionMode.CONTINUOUS,
            window_sec=4.0,
        ),
        OutputFormat.FIF,
    )
    assert results[0].status == "done"
    assert results[0].outputs[0].parent == output_root / "subject"
    manifest = next(output_root.glob("eeg_bg_manifest_*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["items"][0]["status"] == "done"
    assert "diagnostics" in payload["items"][0]


def test_output_must_not_be_inside_input(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    try:
        validate_batch_roots(root, root / "processed")
    except ValueError as exc:
        assert "不能" in str(exc)
    else:
        raise AssertionError("nested output root was accepted")


def test_batch_failure_does_not_block_later_files(synthetic_fif, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    bad = input_root / "bad.fif"
    bad.write_text("not a fif", encoding="utf-8")
    good = input_root / synthetic_fif.name
    good.write_bytes(synthetic_fif.read_bytes())
    results = BatchProcessor().run(
        [bad, good],
        input_root,
        tmp_path / "output",
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(
            mode=ExtractionMode.CONTINUOUS,
            window_sec=4.0,
        ),
        OutputFormat.FIF,
    )
    assert [item.status for item in results] == ["failed", "done"]


def test_batch_artifact_warning_is_diagnostic_only(synthetic_fif, tmp_path, caplog):
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw = mne.io.read_raw_fif(synthetic_fif, preload=True, verbose=False)
    raw._data[0, 100:110] = 900e-6
    source = input_root / "extreme-raw.fif"
    raw.save(source, overwrite=True, verbose=False)
    output_root = tmp_path / "output"
    results = BatchProcessor().run(
        [source],
        input_root,
        output_root,
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(mode=ExtractionMode.CONTINUOUS, window_sec=4.0),
        OutputFormat.FIF,
        artifact_settings=ArtifactSettings(enabled=True, threshold_uv=200.0),
    )
    item = results[0]
    assert item.status == "warning"
    assert item.outputs[0].is_file()
    assert "原始输入超过" in item.warnings[-1]
    diagnostic = item.diagnostics["artifact_threshold"]
    assert diagnostic["affected_channels"] == ["FP1"]
    assert diagnostic["exceedance_region_count"] == 1
    assert "原始输入超过" in caplog.text

    payload = json.loads(next(output_root.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["artifact_settings"] == {"enabled": True, "threshold_uv": 200.0}
    assert payload["items"][0]["diagnostics"]["artifact_threshold"] == diagnostic
    csv_text = next(output_root.glob("*.csv")).read_text(encoding="utf-8-sig")
    assert "artifact_threshold_enabled" in csv_text
    assert "200.0" in csv_text


def test_disabled_batch_artifact_threshold_emits_no_warning(synthetic_fif, tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw = mne.io.read_raw_fif(synthetic_fif, preload=True, verbose=False)
    raw._data[0, 100:110] = 900e-6
    source = input_root / "extreme-raw.fif"
    raw.save(source, overwrite=True, verbose=False)
    results = BatchProcessor().run(
        [source],
        input_root,
        tmp_path / "output",
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(mode=ExtractionMode.CONTINUOUS, window_sec=4.0),
        OutputFormat.FIF,
        artifact_settings=ArtifactSettings(enabled=False, threshold_uv=200.0),
    )
    assert results[0].status == "done"
    assert not any("原始输入超过" in warning for warning in results[0].warnings)
