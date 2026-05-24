import numpy as np
import pandas as pd
from pathlib import Path


def build_subject_index(cfg: dict) -> pd.DataFrame:
    data_root = Path(cfg["paths"]["data_root"])
    montage_dir = cfg["dataset"]["montage_dir"]
    reference = cfg["dataset"]["reference_scheme"]
    classes = cfg["dataset"]["classes"]

    records = []
    for label_name, folder in classes.items():
        label = 0 if label_name == "epilepsy" else 1
        class_dir = data_root / folder
        if not class_dir.exists():
            continue
        for subject_dir in sorted(class_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                montage_path = session_dir / montage_dir
                if not montage_path.exists():
                    continue
                for edf_file in sorted(montage_path.glob("*.edf")):
                    token_id = edf_file.stem.split("_")[-1]
                    records.append({
                        "subject_id": subject_dir.name,
                        "session_id": session_dir.name,
                        "token_id": token_id,
                        "label": label,
                        "reference": reference,
                        "edf_path": str(edf_file),
                    })
    return pd.DataFrame(records)


def assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["split"]["random_seed"])
    train_frac = cfg["split"]["train"]
    val_frac   = cfg["split"]["val"]

    # Stratified by label: split each class independently so val/test always
    # contain subjects from both classes even with a small val fraction (10%).
    subject_label = index.drop_duplicates("subject_id").set_index("subject_id")["label"]
    train_set, val_set = set(), set()
    for _label, group in index.groupby(
        index["subject_id"].map(subject_label)
    ):
        subjects = group["subject_id"].unique().copy()
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * train_frac)
        n_val   = max(1, int(n * val_frac))   # guarantee ≥1 val subject per class
        train_set.update(subjects[:n_train])
        val_set.update(subjects[n_train : n_train + n_val])

    def _split(sid: str) -> str:
        if sid in train_set:
            return "train"
        if sid in val_set:
            return "val"
        return "test"

    index = index.copy()
    index["split"] = index["subject_id"].apply(_split)
    return index
