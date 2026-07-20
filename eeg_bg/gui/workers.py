from __future__ import annotations

from pathlib import Path
import threading

from PySide6.QtCore import QObject, Signal, Slot

from eeg_bg.application.batch import BatchProcessor, scan_recordings
from eeg_bg.application.models import ExtractionSpec, OutputFormat, ProcessingSpec
from eeg_bg.application.processing import ProcessingEngine
from eeg_bg.application.recording import RecordingService
from eeg_bg.exceptions import ProcessingCancelled


class BaseWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)
    cancelled = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = threading.Event()

    def request_cancel(self):
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()


class LoadWorker(BaseWorker):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    @Slot()
    def run(self):
        try:
            if self.is_cancelled():
                raise ProcessingCancelled("用户已取消读取")
            service = RecordingService()
            info = service.inspect(self.path)
            raw, warnings = service.load_eeg(self.path, preload=True)
            if self.is_cancelled():
                raise ProcessingCancelled("用户已取消读取")
            self.finished.emit((info, raw, warnings))
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class ProcessWorker(BaseWorker):
    def __init__(self, path: Path, processing: ProcessingSpec, extraction: ExtractionSpec):
        super().__init__()
        self.path = path
        self.processing = processing
        self.extraction = extraction

    @Slot()
    def run(self):
        try:
            engine = ProcessingEngine()
            result = engine.process(
                self.path,
                self.processing,
                self.extraction,
                cancel_requested=self.is_cancelled,
                progress=lambda a, b, c: self.progress.emit(a, b, c),
            )
            self.finished.emit(result)
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(BaseWorker):
    def __init__(self, jobs: list[tuple[object, Path]], output_format: OutputFormat):
        super().__init__()
        self.jobs = jobs
        self.output_format = output_format

    @Slot()
    def run(self):
        try:
            service = RecordingService()
            written = []
            for index, (raw, path) in enumerate(self.jobs):
                if self.is_cancelled():
                    raise ProcessingCancelled("导出已取消；已完成的文件保持有效")
                service.write(raw, path, self.output_format)
                written.append(path)
                self.progress.emit(index + 1, len(self.jobs), "写出文件")
            self.finished.emit(written)
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class ScanWorker(BaseWorker):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    @Slot()
    def run(self):
        try:
            self.finished.emit(scan_recordings(self.root))
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchWorker(BaseWorker):
    itemStarted = Signal(int, int, str)
    itemFinished = Signal(object)

    def __init__(
        self,
        files: list[Path],
        input_root: Path,
        output_root: Path,
        processing: ProcessingSpec,
        extraction: ExtractionSpec,
        output_format: OutputFormat,
        overwrite: bool,
    ):
        super().__init__()
        self.files = files
        self.input_root = input_root
        self.output_root = output_root
        self.processing = processing
        self.extraction = extraction
        self.output_format = output_format
        self.overwrite = overwrite

    @Slot()
    def run(self):
        try:
            processor = BatchProcessor()
            results = processor.run(
                self.files,
                self.input_root,
                self.output_root,
                self.processing,
                self.extraction,
                self.output_format,
                overwrite=self.overwrite,
                cancel_requested=self.is_cancelled,
                item_progress=lambda a, b, p: self.itemStarted.emit(a, b, str(p)),
                stage_progress=lambda a, b, c: self.progress.emit(a, b, c),
                item_finished=lambda result: self.itemFinished.emit(result),
            )
            if self.is_cancelled():
                self.cancelled.emit("批量任务已取消；清单已写出")
            else:
                self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
