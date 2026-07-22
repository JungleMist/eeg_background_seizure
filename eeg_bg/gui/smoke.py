from __future__ import annotations

from pathlib import Path
import tempfile

import mne
import numpy as np

from eeg_bg.application.models import (
    ExtractionMode,
    ExtractionSpec,
    OutputFormat,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
)
from eeg_bg.application.processing import ProcessingEngine


CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


def _synthetic_raw():
    sfreq = 125.0
    times = np.arange(int(sfreq * 4.0)) / sfreq
    rng = np.random.default_rng(42)
    shared = 25e-6 * np.sin(2 * np.pi * 8 * times)
    data = np.stack([
        shared + rng.normal(scale=3e-6, size=len(times)) for _ in CHANNELS
    ])
    return mne.io.RawArray(
        data,
        mne.create_info(CHANNELS, sfreq, ch_types="eeg"),
        verbose=False,
    )


def run_smoke_test() -> int:
    with tempfile.TemporaryDirectory(prefix="eeg_bg_studio_smoke_") as tmp:
        root = Path(tmp)
        source_fif = root / "synthetic-raw.fif"
        source_edf = root / "synthetic.edf"
        synthetic = _synthetic_raw()
        synthetic.save(source_fif, overwrite=True, verbose=False)
        synthetic.export(source_edf, fmt="edf", overwrite=True, verbose=False)
        extraction = ExtractionSpec(
            mode=ExtractionMode.CONTINUOUS,
            window_sec=4.0,
        )
        engine = ProcessingEngine()
        methods = [
            ProcessingSpec(
                method=ProcessingMethod.BASIC,
                analysis_window_sec=4.0,
            ),
            ProcessingSpec(
                method=ProcessingMethod.ICA,
                analysis_window_sec=4.0,
                ica_n_components=4,
            ),
            *[
                ProcessingSpec(
                    method=ProcessingMethod.WIENER,
                    wiener_mode=mode,
                    analysis_window_sec=4.0,
                    coherence_threshold=0.0,
                )
                for mode in WienerMode
            ],
        ]
        for source in (source_fif, source_edf):
            for spec in methods:
                result = engine.process(source, spec, extraction)
                if result.preview_raw.n_times != 500:
                    raise RuntimeError(
                        f"Unexpected {source.suffix} output length for {spec.method.value}"
                    )
        basic = engine.process(source_fif, methods[0], extraction).preview_raw
        engine.recordings.write(basic, root / "smoke.edf", OutputFormat.EDF)
        engine.recordings.write(basic, root / "smoke.fif", OutputFormat.FIF)
        if not (root / "smoke.edf").is_file() or not (root / "smoke.fif").is_file():
            raise RuntimeError("EDF/FIF smoke exports were not created")
    return 0
