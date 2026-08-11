from copy import deepcopy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two config dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(config_path: Path, seen: set[Path] | None = None) -> dict:
    """Load a YAML file, optionally inheriting from another config."""
    config_path = config_path.resolve()
    seen = seen or set()
    if config_path in seen:
        raise ValueError(f"Circular config extends detected at {config_path}")
    seen.add(config_path)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("extends", None)
    if not base_name:
        return cfg

    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_cfg = _load_yaml(base_path, seen)
    return _deep_merge(base_cfg, cfg)


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)
    cfg.setdefault("wiener", {}).setdefault(
        "protected_band_hz", [5.0, 20.0]
    )
    cfg["wiener"].setdefault("coherent_gate_enabled", True)
    cfg["wiener"].setdefault("coherent_gate_threshold_uv", 100.0)
    project_root = config_path.parent.parent.resolve()
    for key in ("cache_dir", "results_dir"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            if not p.is_absolute():
                cfg["paths"][key] = str(project_root / p)
    # Local experiment configs created before dataset.active used a flat TUEP
    # block. Keep them usable while all newly tracked configs use the nested
    # dataset.tuep / dataset.tuab layout.
    dataset_cfg = cfg.get("dataset", {})
    if "active" not in dataset_cfg and "reference_scheme" in dataset_cfg:
        cfg["dataset"] = {"active": "tuep", "tuep": dataset_cfg}
    return cfg
