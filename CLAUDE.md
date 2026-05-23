# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python via **conda env `eeg_pipeline`** (`C:\ProgramData\anaconda3\envs\eeg_pipeline`). Use `conda run -n eeg_pipeline` or activate it before running commands. The conda base env has NumPy 2.x which is incompatible with the pinned scipy/matplotlib.
- Install the package in development mode: `pip install -e .`
- Core dependencies: numpy 1.26.4, scipy 1.17.1, mne 1.11.0, scikit-learn 1.8.0, pandas, matplotlib, pyyaml, pytest, tqdm.

## Common Commands

```bash
# Install
conda run -n eeg_pipeline pip install -e .

# Run all unit tests (no real EDF data required)
conda run -n eeg_pipeline python -m pytest tests/ -v

# Skip integration tests (default — no real EDF needed)
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration"

# Run a single test module
conda run -n eeg_pipeline python -m pytest tests/test_decomposition/test_wiener.py -v

# Run a single test by name
conda run -n eeg_pipeline python -m pytest tests/test_decomposition/test_wiener.py::test_decompose_epoch_reduces_coherence -v
```

## Architecture

The package is `eeg_bg/`, pip-installed from `setup.py`. Configuration is loaded from `configs/default.yaml` via `eeg_bg.config.settings.load_config()`, which resolves relative `cache_dir` / `results_dir` paths against the project root.

**Data flow (unidirectional):**
```
EDF files (TUEP v3.1.0, D:/EEGdata/TUEP/v3.1.0)
  → io/edf_reader.py      — loads EDF, normalises channel names (strips "EEG " prefix and "-REF" suffix), resamples to 125 Hz, converts V→µV
  → io/annotation.py      — parses csv_bi files; excludes ±30 s seizure buffers from background intervals
  → preprocessing/epoch.py — slices 8 s epochs, rejects epochs exceeding 200 µV
  → cache/epochs/          — checkpoint 1 (.npz via io/cache.py)
  → decomposition/wiener.py — vector Wiener decomposition (core method)
  → decomposition/ica.py    — FastICA ablation baseline
  → cache/wiener|ica/       — checkpoint 2
  → verification/           — V1 coherence, V2 transitivity, V3 frequency variation
  → visualization/          — matplotlib figures returned as plt.Figure (never call plt.show())
```

### Key modules

| Module | Responsibility |
|--------|---------------|
| `eeg_bg/decomposition/wiener.py` | Core Wiener filter: `estimate_cross_psd` → `compute_wiener_filter` → `apply_wiener_filter` → `decompose_epoch` / `decompose_subject`. Returns `WienerResult` (raw, specific, coherent, filters dict, skipped_pairs). |
| `eeg_bg/decomposition/wiener_scalar.py` | Fixed-scalar ablation baseline for comparison. |
| `eeg_bg/verification/coherence.py` | V1: pairwise coherence matrix before/after decomposition (`run_v1`). Uses `freq_resolution_hz` (not `nperseg`) to set estimation window so coherence averaging is valid. |
| `eeg_bg/verification/transitivity.py` | V2: single-point-source transitivity constraint; V3: frequency variation of \|h(f)\| across target band. |
| `eeg_bg/visualization/psd_plots.py` | `plot_psd_comparison`: PSD overlay (raw / Wiener-specific / ICA-cleaned) for target channels. Channels default to `cfg["visualization"]["psd_target_channels"]` (FP1, FP2). Uses boxcar Welch consistent with decomposition. |
| `eeg_bg/io/dataset.py` | Traverses TUEP directory tree → subject index DataFrame; `assign_splits` splits by subject (not recording). |
| `eeg_bg/io/cache.py` | `load_or_compute` wraps any `compute_fn` with `.npz` on-disk caching; cache key = SHA-256 of `edf_path|start_sec|sfreq|bandpass`. |

### Channel groups (G1–G6)

The Wiener filter operates on **movement-artifact conduction pathways**, not bilateral pairs. Groups are defined in `configs/default.yaml` under `channels.channel_groups`:
- G1 `[FP1, FP2]` — symmetric facial (frontalis)
- G2 `[F7, T3]` — left SCM
- G3 `[T3, T5, O1]` — left posterior neck (3-channel chain)
- G4 `[O1, O2]` — bilateral occipitalis
- G5 `[F8, T4]` — right SCM
- G6 `[T4, T6, O2]` — right posterior neck (3-channel chain)

Passthrough channels (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz`) are never filtered.

### Wiener filter implementation details

- PSD estimated with **boxcar window** so that when `nperseg == n_times` the filter can be applied exactly via rfft without windowing mismatch.
- When `nperseg < n_times`, filter coefficients are linearly interpolated to the full rfft grid; `specific + coherent == raw` is guaranteed by construction regardless.
- A coherence gate (max pairwise coherence across all pairs in the group, over the target frequency band) skips groups below `coherence_threshold` (default 0.15).
- `nperseg` in `wiener:` is for filter estimation; V1 coherence uses `freq_resolution_hz` (125 / 0.5 = 250 samples = 4 segments per 1000-sample epoch) to avoid trivial coherence=1.

### Output directory structure

```
results/
├── figures/
│   └── {subject_id}/
│       ├── waveform_comparison.png   — all-channel stacked waveform (raw | Wiener | ICA)
│       └── psd_comparison.png        — PSD overlay for psd_target_channels
└── verification/
    ├── v1_coherence.csv
    ├── v2_transitivity.csv
    └── v3_frequency_variation.csv
```

Script 05 accepts `--channels "FP1,FP2,T3"` to override `psd_target_channels` at runtime.

## Tests

All tests run without real EDF data. `tests/conftest.py` provides:
- `synthetic_epoch`: 19-ch, 1000-sample epoch — single broadband point source mixed with known gains + independent noise (high SNR: source 50 µV, noise 1 µV).
- `synthetic_epochs_batch`: batch of 5 epochs.
- `tmp_cache_dir`: temporary directory for cache tests.

Integration tests (requiring real TUEP EDF files) should be marked `@pytest.mark.integration`.
