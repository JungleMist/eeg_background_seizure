"""Application services shared by the eeg_bg Studio GUI and batch runner."""

from .ern_comparison import (
    ErnComparisonResult,
    ErnComparisonService,
    ErnWaveform,
    ResponseTrial,
    build_response_trials,
)
from .models import (
    ArtifactSettings,
    BatchItemResult,
    ExtractionMode,
    ExtractionSpec,
    OutputFormat,
    ProcessingMethod,
    ProcessingResult,
    ProcessingSpec,
    RecordingEvent,
    RecordingSidecars,
    WienerMode,
    pipeline_fingerprint,
)
from .processing import ProcessingEngine
from .recording import RecordingService

__all__ = [
    "ArtifactSettings",
    "BatchItemResult",
    "ErnComparisonResult",
    "ErnComparisonService",
    "ErnWaveform",
    "ExtractionMode",
    "ExtractionSpec",
    "OutputFormat",
    "ProcessingEngine",
    "ProcessingMethod",
    "ProcessingResult",
    "ProcessingSpec",
    "RecordingEvent",
    "RecordingService",
    "RecordingSidecars",
    "ResponseTrial",
    "WienerMode",
    "build_response_trials",
    "pipeline_fingerprint",
]
