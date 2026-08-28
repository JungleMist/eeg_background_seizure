"""Tests for Script 19's TUAB component EEGNet training path."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml


SCRIPT_PATH = Path("scripts/19_train_tuab_component_eegnet.py")
STANDARD_19 = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("script19", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


def _write_config(tmp_path: Path, **overrides) -> Path:
    model = {
        "random_state": 42,
        "device": "cpu",
        "num_workers": 0,
        "F1": 2,
        "D": 1,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 2,
        "records_per_step": 2,
        "max_epochs": 1,
        "patience": 1,
        "lr_patience": 1,
        "lr_factor": 0.5,
    }
    model.update(overrides)
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
        },
        "dataset": {"active": "tuab"},
        "channels": {"standard_19": list(STANDARD_19)},
        "preprocessing": {"target_sfreq": 125, "epoch_length_sec": 20},
        "wiener": {"mode": "frequency"},
        "ml": {"tuab_component_eegnet": model},
    }
    config_path = tmp_path / "configs" / "tuab_test.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_record(
    root: Path,
    evaluation_id: str,
    patient_id: str,
    split: str,
    label: int,
    n_epochs: int,
    *, source_mode: str = "frequency",
) -> Path:
    rng = np.random.default_rng(abs(hash(evaluation_id)) % (2**32))
    raw = rng.normal(size=(n_epochs, 19, 2500)).astype(np.float32)
    coherent = (raw * np.float32(0.25)).astype(np.float32)
    coherent[:, -1] = 0.0
    specific = (raw - coherent).astype(np.float32)
    specific_coherent = np.concatenate([specific, coherent], axis=1)
    combined_names = (
        *(f"specific::{channel}" for channel in STANDARD_19),
        *(f"coherent::{channel}" for channel in STANDARD_19),
    )
    starts = np.arange(n_epochs, dtype=np.int64) * 2500
    path = root / f"{evaluation_id}.npz"
    np.savez_compressed(
        path,
        raw=raw,
        specific=specific,
        coherent=coherent,
        specific_coherent=specific_coherent,
        epoch_start_samples=starts,
        epoch_start_sec=starts.astype(np.float64) / 125.0,
        label=np.asarray(label, dtype=np.int8),
        class_name=np.asarray("normal" if label else "abnormal"),
        split=np.asarray(split),
        patient_id=np.asarray(patient_id),
        recording_id=np.asarray(evaluation_id),
        evaluation_id=np.asarray(evaluation_id),
        subject_id=np.asarray(evaluation_id),
        ch_names=np.asarray(STANDARD_19, dtype="U"),
        specific_coherent_ch_names=np.asarray(combined_names, dtype="U"),
        sfreq=np.asarray(125.0),
        epoch_samples=np.asarray(2500),
        n_epochs=np.asarray(n_epochs),
        source_mode=np.asarray(source_mode),
        fingerprint=np.asarray(f"fingerprint-{evaluation_id}"),
        schema_version=np.asarray(2),
    )
    return path


def _write_input_tree(tmp_path: Path, *, leak: bool = False) -> Path:
    root = tmp_path / "epochs"
    root.mkdir(parents=True)
    definitions = [
        ("train_abn", "patient_train_abn", "train", 0, 1),
        ("train_norm", "patient_train_norm", "train", 1, 3),
        ("val_abn", "patient_val_abn", "val", 0, 2),
        ("val_norm", "patient_val_norm", "val", 1, 1),
        ("test_abn", "patient_test_abn", "test", 0, 1),
        ("test_norm", "patient_test_norm", "test", 1, 2),
    ]
    if leak:
        definitions[2] = (
            "val_abn", "patient_train_abn", "val", 0, 2
        )
    rows = []
    for evaluation_id, patient_id, split, label, n_epochs in definitions:
        path = _write_record(
            root, evaluation_id, patient_id, split, label, n_epochs
        )
        rows.append({
            "evaluation_id": evaluation_id,
            "status": "processed",
            "output_path": str(path),
        })
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump({
            "dataset": {"active": "tuab"},
            "wiener": {"mode": "frequency"},
        }),
        encoding="utf-8",
    )
    return root


class TinyEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        signal = tensors.mean(dim=(1, 2, 3)) * 0.01
        return torch.sigmoid(signal + self.bias).unsqueeze(1)


def test_discovers_script18_records_and_maps_validation_split(module, tmp_path):
    config_path = _write_config(tmp_path)
    input_root = _write_input_tree(tmp_path)
    cfg = module.load_config(config_path)

    records = module.discover_records(input_root, cfg, "frequency")

    assert len(records) == 6
    assert {record["split"] for record in records} == {
        "train", "validation", "test"
    }
    assert {record["label"] for record in records} == {0, 1}
    assert all(record["epoch_samples"] == 2500 for record in records)
    assert all(
        len(record["specific_coherent_ch_names"]) == 38 for record in records
    )


def test_rejects_patient_leakage(module, tmp_path):
    config_path = _write_config(tmp_path)
    input_root = _write_input_tree(tmp_path, leak=True)
    cfg = module.load_config(config_path)

    with pytest.raises(ValueError, match="patient leakage"):
        module.discover_records(input_root, cfg, "frequency")


def test_rejects_single_class_split(module, tmp_path):
    config_path = _write_config(tmp_path)
    input_root = _write_input_tree(tmp_path)
    (input_root / "val_norm.npz").unlink()
    manifest = pd.read_csv(input_root / "manifest.csv")
    manifest.loc[manifest["evaluation_id"] == "val_norm", "status"] = "skipped"
    manifest.to_csv(input_root / "manifest.csv", index=False)

    with pytest.raises(ValueError, match="validation split must contain"):
        module.discover_records(
            input_root, module.load_config(config_path), "frequency"
        )


def test_epoch_channel_zscore_preserves_zero_channels(module):
    values = np.zeros((2, 3, 8), dtype=np.float32)
    values[:, 0] = np.arange(8, dtype=np.float32)

    normalized = module.trial_channel_zscore(values)

    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized[:, 1:], 0.0)
    np.testing.assert_allclose(normalized[:, 0].mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[:, 0].std(axis=-1), 1.0, atol=1e-6)


def test_record_class_weights_ignore_epoch_counts(module):
    records = [
        {"label": 0, "n_epochs": 1},
        {"label": 0, "n_epochs": 20},
        {"label": 1, "n_epochs": 100},
    ]
    assert module.record_class_weights(records) == {0: 1.0, 1: 2.0}


def test_record_mean_loss_is_independent_of_epoch_count(module):
    model = TinyEEGNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = [
        {"epochs": torch.zeros(1, 19, 8), "label": 0},
        {"epochs": torch.zeros(5, 19, 8), "label": 1},
    ]

    loss = module._train_epoch(
        model,
        loader,
        optimizer,
        nn.BCELoss(reduction="none"),
        torch.device("cpu"),
        {0: 1.0, 1: 1.0},
        epoch_batch_size=2,
        records_per_step=2,
    )

    assert loss == pytest.approx(np.log(2.0), rel=1e-6)


def test_device_resolution_and_explicit_failure(module, monkeypatch):
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.backends.mps, "is_available", lambda: False)
    assert module.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        module.resolve_device("mps")


def test_eegnet_accepts_tuab_19_by_2500_input(module):
    model = module._make_model(
        19,
        2500,
        {"F1": 8, "D": 2, "dropout": 0.25},
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 19, 2500))
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_eegnet_accepts_tuab_38_by_2500_input(module):
    model = module._make_model(
        38,
        2500,
        {"F1": 8, "D": 2, "dropout": 0.25},
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 38, 2500))
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_training_outputs_cache_reuse_and_force(
    module, tmp_path, monkeypatch
):
    config_path = _write_config(tmp_path)
    input_root = _write_input_tree(tmp_path)
    output_root = tmp_path / "training"
    monkeypatch.setattr(module, "_make_model", lambda *_: TinyEEGNet())

    result = module.run(
        config_path,
        input_dir=input_root,
        output_dir=output_root,
        condition="all",
        device="cpu",
    )

    condition_dir = result / "conditions" / "raw"
    assert all(
        (result / "conditions" / condition / "cache.json").is_file()
        for condition in module.CONDITIONS
    )
    assert (condition_dir / "best_model.pt").is_file()
    assert (condition_dir / "cache.json").is_file()
    assert (result / "condition_metrics.csv").is_file()
    params = json.loads((condition_dir / "best_params.json").read_text())
    assert params["checkpoint_selection"] == "validation_recording_auprc"
    assert params["threshold_selection"] == (
        "validation_recording_balanced_accuracy"
    )
    combined_params = json.loads(
        (
            result / "conditions" / "specific_coherent" / "best_params.json"
        ).read_text()
    )
    assert combined_params["n_channels"] == 38
    predictions = pd.read_csv(condition_dir / "test_predictions.csv")
    assert len(predictions) == 2
    assert set(predictions["evaluation_id"]) == {"test_abn", "test_norm"}
    assert set(predictions["true_label"]) == {0, 1}

    monkeypatch.setattr(
        module,
        "train_condition",
        lambda *args, **kwargs: pytest.fail("matching cache should be reused"),
    )
    module.run(
        config_path,
        input_dir=input_root,
        output_dir=output_root,
        condition="raw",
        device="cpu",
    )

    with pytest.raises(module.CacheMismatch, match="--force"):
        module.run(
            config_path,
            input_dir=input_root,
            output_dir=output_root,
            condition="raw",
            device="cpu",
            random_state=99,
        )


def test_test_predictions_are_deferred_until_validation_is_frozen(
    module, tmp_path, monkeypatch
):
    records = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            records.append({
                "path": tmp_path / f"{split}_{label}.npz",
                "evaluation_id": f"{split}_{label}",
                "patient_id": f"patient_{split}_{label}",
                "recording_id": f"{split}_{label}",
                "label": label,
                "class_name": "normal" if label else "abnormal",
                "split": split,
                "n_epochs": 1,
                "ch_names": STANDARD_19,
                "epoch_samples": 2500,
            })
    model_cfg = {
        "F1": 2, "D": 1, "dropout": 0.0, "lr": 0.001,
        "weight_decay": 0.0, "lr_factor": 0.5, "lr_patience": 1,
        "batch_size": 2, "records_per_step": 2, "max_epochs": 1,
        "patience": 1,
    }
    calls = []

    def fake_predict(_model, selected, condition, *_args, **_kwargs):
        split = selected[0]["split"]
        calls.append(split)
        frame = pd.DataFrame({
            "condition": [condition, condition],
            "split": [split, split],
            "evaluation_id": [record["evaluation_id"] for record in selected],
            "subject_id": [record["evaluation_id"] for record in selected],
            "patient_id": [record["patient_id"] for record in selected],
            "recording_id": [record["recording_id"] for record in selected],
            "n_epochs": [1, 1],
            "true_label": [0, 1],
            "pred_proba": [0.2, 0.8],
        })
        epoch_frame = frame.drop(columns=["subject_id", "n_epochs"]).copy()
        epoch_frame["epoch_index"] = [0, 0]
        return frame, epoch_frame

    monkeypatch.setattr(module, "_make_model", lambda *_: TinyEEGNet())
    monkeypatch.setattr(module, "_train_epoch", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(module, "_predict_records", fake_predict)

    module.train_condition(
        "raw", records, model_cfg, tmp_path / "condition", 43, "cpu", 0,
        "test-fingerprint",
    )

    assert calls == ["validation", "validation", "test"]


def test_partial_condition_cache_requires_force(module, tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    input_root = _write_input_tree(tmp_path)
    output_root = tmp_path / "training"
    monkeypatch.setattr(module, "_make_model", lambda *_: TinyEEGNet())
    module.run(
        config_path,
        input_dir=input_root,
        output_dir=output_root,
        condition="coherent",
        device="cpu",
    )
    (output_root / "conditions" / "coherent" / "history.csv").unlink()

    with pytest.raises(module.CacheMismatch, match="corrupt"):
        module.run(
            config_path,
            input_dir=input_root,
            output_dir=output_root,
            condition="coherent",
            device="cpu",
        )

    rebuilt = module.run(
        config_path,
        input_dir=input_root,
        output_dir=output_root,
        condition="coherent",
        device="cpu",
        force=True,
    )
    assert (
        rebuilt / "conditions" / "coherent" / "history.csv"
    ).is_file()
