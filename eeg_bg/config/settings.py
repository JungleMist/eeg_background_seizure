from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent.resolve()
    for key in ("cache_dir", "results_dir"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            if not p.is_absolute():
                cfg["paths"][key] = str(project_root / p)
    return cfg
