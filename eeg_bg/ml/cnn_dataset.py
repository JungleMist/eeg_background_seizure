"""PyTorch Dataset that reads epoch caches for CNN training.

Mirrors the interface of ``build_dataset()`` in ``eeg_bg/features/extraction.py``
but returns raw epoch tensors instead of hand-crafted feature vectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Cache subdirectory names keyed by condition label — same mapping as
# eeg_bg/features/extraction.py::_CONDITION_TO_SUBDIR
_CONDITION_TO_SUBDIR: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "wiener_frequency",
    "ica":    "ica",
}

# NPZ array key holding signal data per condition
_CONDITION_TO_KEY: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "specific",
    "ica":    "specific",
}


class EEGEpochDataset(Dataset):
    """Dataset of EEG epochs loaded from the project's NPZ cache.

    Each item is a tuple ``(epoch_tensor, label, subject_id)`` where
    ``epoch_tensor`` has shape ``(1, 19, 1000)`` and is z-scored per channel.

    Parameters
    ----------
    cache_root : Path
        Project-level cache directory (resolved absolute path from ``load_config``).
    condition : str
        One of ``"raw"``, ``"wiener"``, ``"ica"``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Raises
    ------
    ValueError
        If *condition* is not recognised.
    FileNotFoundError
        If the condition subdirectory does not exist.
    """

    def __init__(
        self,
        cache_root: Path,
        condition: str,
        split: str,
    ) -> None:
        if condition not in _CONDITION_TO_SUBDIR:
            raise ValueError(
                f"Unknown condition {condition!r}. "
                f"Expected one of {list(_CONDITION_TO_SUBDIR)}."
            )

        subdir = Path(cache_root) / _CONDITION_TO_SUBDIR[condition]
        if not subdir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {subdir}. "
                f"Run the corresponding preprocessing script first."
            )

        array_key = _CONDITION_TO_KEY[condition]

        # Load all matching epochs eagerly into memory.
        # Epoch arrays are small (19 × 1000 × 4 bytes ≈ 76 KB each) and the
        # full dataset typically fits comfortably in RAM.
        self._epochs: list[np.ndarray] = []      # each (19, 1000) float32
        self._labels: list[int] = []
        self._subject_ids: list[str] = []

        for npz_path in sorted(subdir.rglob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            if str(data["split"]) != split:
                continue

            epochs_arr = data[array_key].astype(np.float32)  # (n_ep, 19, 1000)
            label      = int(data["label"])
            subject_id = str(data["subject_id"])

            for ep in epochs_arr:
                self._epochs.append(ep)
                self._labels.append(label)
                self._subject_ids.append(subject_id)

    def __len__(self) -> int:
        return len(self._epochs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        epoch = self._epochs[idx].copy()  # (19, 1000)

        # Z-score each channel independently across the 1000 time samples.
        mean = epoch.mean(axis=-1, keepdims=True)          # (19, 1)
        std  = epoch.std(axis=-1, keepdims=True) + 1e-8   # (19, 1), avoid /0
        epoch = (epoch - mean) / std

        # Unsqueeze to (1, 19, 1000) — single "colour channel" for Conv2d
        tensor = torch.from_numpy(epoch).unsqueeze(0)  # (1, 19, 1000)
        return tensor, self._labels[idx], self._subject_ids[idx]
