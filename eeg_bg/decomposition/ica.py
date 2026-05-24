from __future__ import annotations

import mne
import numpy as np


def fit_ica(
    epochs: np.ndarray,    # (n_epochs, n_ch, n_times) in uV
    ch_names: list[str],
    cfg: dict,
) -> tuple[mne.preprocessing.ICA, list[int]]:
    n_epochs, n_ch, n_times = epochs.shape
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    n_components = cfg["ica"]["n_components"]
    threshold = cfg["ica"]["artifact_corr_threshold"]
    random_state = cfg["ica"].get("random_state", 42)

    # Concatenate epochs: (n_ch, n_epochs * n_times)
    data_2d = epochs.transpose(1, 0, 2).reshape(n_ch, -1) / 1e6  # uV -> V

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",
                           verbose=False)
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica",
        random_state=random_state, verbose=False
    )
    # MNE recommends ≥1 Hz high-pass for ICA convergence. Create a filtered copy
    # used only for fitting; apply_ica runs on the original 0.5 Hz bandpass data.
    raw_for_ica = raw.copy().filter(
        l_freq=1.0, h_freq=None,
        method="iir", iir_params=dict(order=5, ftype="butter"),
        verbose=False,
    )
    ica.fit(raw_for_ica, verbose=False)

    # Artifact detection: correlate each component with frontal proxy (EOG)
    sources = ica.get_sources(raw).get_data()  # (n_components, n_samples)
    frontal = [i for i, ch in enumerate(ch_names) if ch in ("FP1", "FP2")]
    artifact_indices: list[int] = []
    if frontal:
        proxy = data_2d[frontal].mean(axis=0)
        for comp_idx in range(sources.shape[0]):
            corr = np.corrcoef(sources[comp_idx], proxy)[0, 1]
            if abs(corr) > threshold:
                artifact_indices.append(comp_idx)

    return ica, artifact_indices


def apply_ica(
    epochs: np.ndarray,                # (n_epochs, n_ch, n_times) in uV
    ica: mne.preprocessing.ICA,
    artifact_indices: list[int],
    ch_names: list[str],
    cfg: dict,
) -> np.ndarray:
    n_epochs, n_ch, n_times = epochs.shape
    sfreq = float(cfg["preprocessing"]["target_sfreq"])

    data_2d = epochs.transpose(1, 0, 2).reshape(n_ch, -1) / 1e6  # uV -> V
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",
                           verbose=False)
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    ica_copy = ica.copy()
    ica_copy.exclude = artifact_indices
    raw_clean = ica_copy.apply(raw.copy(), verbose=False)

    clean_2d = raw_clean.get_data() * 1e6  # V -> uV
    return clean_2d.reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)
