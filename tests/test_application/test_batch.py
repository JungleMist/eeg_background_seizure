import json
from pathlib import Path

from eeg_bg.application.batch import BatchProcessor, scan_recordings, validate_batch_roots
from eeg_bg.application.models import (
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
    (nested / "ignore.csv").write_text("x", encoding="utf-8")
    found = scan_recordings(tmp_path)
    assert synthetic_fif in found
    assert copied in found
    assert all(path.suffix in {".fif", ".edf"} for path in found)


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
            artifact_threshold_uv=500.0,
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
            artifact_threshold_uv=500.0,
        ),
        OutputFormat.FIF,
    )
    assert [item.status for item in results] == ["failed", "done"]
