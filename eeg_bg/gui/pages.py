from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from eeg_bg.application.batch import output_name
from eeg_bg.application.models import OutputFormat, ProcessingMethod, WienerMode
from eeg_bg.application.processing import ProcessingEngine

from .branding import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION
from .channel_groups import ChannelGroupPresetStore
from .ern_comparison import ErnComparisonDialog
from .events import RecordingInspectorPanel
from .parameters import ArtifactSettingsStore, ParameterPanel
from .waveform import WaveformView
from .workers import (
    BatchWorker,
    ErnComparisonWorker,
    ExportWorker,
    LoadWorker,
    ProcessWorker,
    ScanWorker,
)


def _output_format_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("EDF", OutputFormat.EDF)
    combo.addItem("FIF", OutputFormat.FIF)
    return combo


class ThreadedPage(QWidget):
    statusChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker = None

    def _start_worker(self, worker, on_finished, on_failed=None):
        if self._thread is not None:
            QMessageBox.information(self, "任务正在运行", "请等待当前任务完成或先取消。")
            return False
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.finished.connect(thread.quit)
        worker.failed.connect(on_failed or self._task_failed)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(self._task_cancelled)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()
        return True

    def _thread_finished(self):
        self._thread = None
        self._worker = None

    def request_cancel(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.statusChanged.emit("正在等待当前算法步骤安全停止…")

    def _task_failed(self, message: str):
        self.statusChanged.emit("任务失败")
        QMessageBox.critical(self, "处理失败", message)

    def _task_cancelled(self, message: str):
        self.statusChanged.emit(message or "任务已取消")


class PreviewPage(ThreadedPage):
    def __init__(
        self,
        artifact_store: ArtifactSettingsStore | None = None,
        settings: QSettings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings or QSettings(
            SETTINGS_ORGANIZATION, SETTINGS_APPLICATION
        )
        self.artifact_store = artifact_store or ArtifactSettingsStore(self.settings, self)
        self.channel_group_presets = ChannelGroupPresetStore(self.settings, self)
        self.source_path: Path | None = None
        self.recording_info = None
        self.current_result = None
        self._ern_dialog: ErnComparisonDialog | None = None
        self._default_channel_groups = tuple(
            tuple(group)
            for group in ProcessingEngine().base_cfg["channels"]["channel_groups"]
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("交互预览")
        title.setObjectName("PageTitle")
        subtitle = QLabel("在同一时间轴上核对原始 EEG 与 ECMAD 降噪结果")
        subtitle.setObjectName("Muted")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.open_button = QPushButton("打开 EEG 文件")
        self.open_button.setObjectName("Primary")
        self.open_button.clicked.connect(self.choose_file)
        header.addWidget(self.open_button)
        root.addLayout(header)

        self.metadata = QLabel(
            "尚未打开文件 · 支持 EDF / FIF / EEGLAB SET（SET 需与 FDT 同目录）"
        )
        self.metadata.setObjectName("Metadata")
        self.metadata.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.metadata)

        splitter = QSplitter(Qt.Horizontal)
        plot_panel = QFrame()
        plot_panel.setObjectName("Panel")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(12, 12, 12, 12)
        self.waveform = WaveformView()
        self.waveform.set_artifact_settings(self.artifact_store.snapshot())
        self.artifact_store.changed.connect(self.waveform.set_artifact_settings)
        self.waveform.selectionChanged.connect(self._selection_changed)
        plot_layout.addWidget(self.waveform)
        splitter.addWidget(plot_panel)

        controls_host = QWidget()
        controls_host.setObjectName("ScrollContents")
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.recording_inspector = RecordingInspectorPanel()
        self.recording_inspector.eventActivated.connect(self._event_activated)
        controls_layout.addWidget(self.recording_inspector)
        self.parameters = ParameterPanel(
            allow_selection=True,
            artifact_store=self.artifact_store,
            channel_group_presets=self.channel_group_presets,
        )
        self.parameters.bind_settings(self.settings, "preview/parameters")
        self.parameters.currentWindowRequested.connect(self._use_current_window)
        controls_layout.addWidget(self.parameters)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("导出格式"))
        self.output_format = _output_format_combo()
        output_row.addWidget(self.output_format, 1)
        controls_layout.addLayout(output_row)
        self.process_button = QPushButton("运行预处理")
        self.process_button.setObjectName("Primary")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process)
        self.export_button = QPushButton("导出当前结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        self.ern_compare_button = QPushButton("打开 ERN 三方法叠加")
        self.ern_compare_button.setEnabled(False)
        self.ern_compare_button.setToolTip(
            "使用当前参数计算 Raw、ICA 与 ECMAD 的 FCz 响应锁定 ERP"
        )
        self.ern_compare_button.clicked.connect(self.compare_ern)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.request_cancel)
        controls_layout.addWidget(self.process_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.ern_compare_button)
        controls_layout.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_host)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(410)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 350])
        root.addWidget(splitter, 1)

    def choose_file(self):
        start = str(self.settings.value("preview/last_dir", ""))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 EEG 文件",
            start,
            "EEG 文件 (*.edf *.fif *.fif.gz *.set);;"
            "EEGLAB SET (*.set);;EDF (*.edf);;FIF (*.fif *.fif.gz)",
        )
        if not path:
            return
        self.settings.setValue("preview/last_dir", str(Path(path).parent))
        self._load(Path(path))

    def _set_busy(self, busy: bool):
        self.open_button.setEnabled(not busy)
        self.process_button.setEnabled(not busy and self.source_path is not None)
        self.export_button.setEnabled(not busy and self.current_result is not None)
        self.ern_compare_button.setEnabled(not busy and self._can_compare_ern())
        self.cancel_button.setEnabled(busy)
        self.parameters.setEnabled(not busy)
        self.waveform.set_selection_enabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)

    def _thread_finished(self):
        super()._thread_finished()
        self._set_busy(False)

    def _load(self, path: Path):
        self._set_busy(True)
        self.statusChanged.emit(f"正在读取 {path.name}…")
        worker = LoadWorker(path)
        self._start_worker(worker, self._loaded, self._preview_failed)

    def _loaded(self, payload):
        info, raw, warnings = payload
        self.source_path = info.path
        self.recording_info = info
        self.current_result = None
        if self._ern_dialog is not None:
            self._ern_dialog.close()
            self._ern_dialog = None
        self.waveform.set_original(raw)
        self.parameters.set_current_window_available(True)
        self.parameters.set_channel_context(
            info.ch_names, self._default_channel_groups
        )
        self.recording_inspector.set_recording(info)
        self.parameters.stop_sec.setMaximum(info.duration_sec)
        self.parameters.stop_sec.setValue(min(20.0, info.duration_sec))
        self.metadata.setText(
            f"{info.format.upper()}   ·   {info.path.name}   ·   "
            f"{len(info.ch_names)} ch   ·   "
            f"{info.sfreq:.1f} Hz   ·   {info.duration_sec:.2f} s"
        )
        self.export_button.setEnabled(False)
        message = "文件已就绪"
        if warnings:
            message += " · " + "；".join(warnings)
        self.statusChanged.emit(message)

    def _can_compare_ern(self) -> bool:
        info = self.recording_info
        if info is None or "FCz" not in info.ch_names:
            return False
        task_name = str(info.sidecars.eeg.get("TaskName", "")).upper()
        filename_is_ern = "_task-ern_" in info.path.name.lower()
        return bool(info.sidecars.events) and (task_name == "ERN" or filename_is_ern)

    def compare_ern(self):
        if self.source_path is None or not self._can_compare_ern():
            QMessageBox.information(
                self,
                "当前记录不能进行 ERN 对比",
                "请选择包含 FCz、ERN 任务元数据和事件标记的 ERP-CORE SET 文件。",
            )
            return
        try:
            processing = self.parameters.processing_spec()
            processing.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "参数需要调整", str(exc))
            return
        self._set_busy(True)
        self.statusChanged.emit("正在计算 Raw / ICA / ECMAD 响应锁定 ERN…")
        worker = ErnComparisonWorker(self.source_path, processing)
        worker.progress.connect(self._progress_changed)
        self._start_worker(worker, self._ern_ready, self._preview_failed)

    def _ern_ready(self, result):
        if self._ern_dialog is not None:
            self._ern_dialog.close()
        dialog = ErnComparisonDialog(result, self)
        dialog.destroyed.connect(
            lambda _object=None, target=dialog: self._ern_dialog_closed(target)
        )
        self._ern_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.statusChanged.emit(
            f"ERN 对比已就绪 · {result.n_epochs} 试次 · "
            f"错误 {result.n_incorrect} / 正确 {result.n_correct}"
        )

    def _ern_dialog_closed(self, dialog):
        if self._ern_dialog is dialog:
            self._ern_dialog = None

    def _event_activated(self, event):
        self.waveform.focus_event(event)
        self.statusChanged.emit(
            f"已定位 {event.trial_type or 'event'} / {event.value or 'n/a'}"
            f" · {event.onset_sec:.4f} s"
        )

    def _selection_changed(self, start: float, stop: float):
        self.parameters.set_selection(start, stop)

    def _use_current_window(self):
        visible_range = self.waveform.visible_time_range()
        if visible_range is None:
            return
        start, stop = visible_range
        self.waveform.set_selection(start, stop)
        self.parameters.set_selection(start, stop)
        self.statusChanged.emit(
            f"已使用当前窗口作为选区 · {start:.2f}–{stop:.2f} s"
        )

    def process(self):
        if self.source_path is None:
            return
        try:
            processing = self.parameters.processing_spec()
            extraction = self.parameters.extraction_spec()
            processing.validate()
            extraction.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "参数需要调整", str(exc))
            return
        self._set_busy(True)
        self.statusChanged.emit("正在预处理 EEG…")
        worker = ProcessWorker(self.source_path, processing, extraction)
        worker.progress.connect(self._progress_changed)
        self._start_worker(worker, self._processed, self._preview_failed)

    def _progress_changed(self, current: int, total: int, label: str):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.statusChanged.emit(f"{label} {current}/{total}")

    def _processed(self, result):
        self.current_result = result
        segment = result.processed_segments[0]
        label = self._processing_label(result.processing_spec)
        self.waveform.set_processed(segment.raw, segment.start_sec, label)
        status = f"处理完成 · {label}"
        if result.warnings:
            status += " · ⚠ " + "；".join(result.warnings)
        self.statusChanged.emit(status)

    @staticmethod
    def _processing_label(processing) -> str:
        band = (
            f"{processing.bandpass_low_hz:g}–{processing.bandpass_high_hz:g} Hz"
            f" / {processing.target_sfreq:g} Hz"
        )
        if processing.method == ProcessingMethod.BASIC:
            return f"基础处理 · {band}"
        if processing.method == ProcessingMethod.ICA:
            components = processing.ica_n_components or "自动"
            return (
                f"ICA · 组件 {components} · "
                f"阈值 {processing.ica_artifact_corr_threshold:g} · {band}"
            )
        label = (
            f"ECMAD {processing.wiener_mode.value} · "
            f"coherence {processing.coherence_threshold:g}"
        )
        label += (
            f" · coherent 门控 {processing.coherent_gate_threshold_uv:g} µV"
            if processing.coherent_gate_enabled
            else " · coherent 门控关闭"
        )
        if processing.channel_groups is not None:
            label += f" · 导联组 {len(processing.channel_groups)}"
        if processing.wiener_mode != WienerMode.FREQUENCY:
            label += f" · gate {processing.phase_gate_threshold_rad:g} rad"
        if processing.protected_band_hz is None:
            label += " · 保护频带关闭"
        else:
            low_hz, high_hz = processing.protected_band_hz
            label += f" · 保护 {low_hz:g}–{high_hz:g} Hz"
        return f"{label} · {band}"

    def _preview_failed(self, message: str):
        self._task_failed(message)

    def _task_cancelled(self, message: str):
        super()._task_cancelled(message)

    def export_result(self):
        if self.current_result is None:
            return
        fmt = OutputFormat(self.output_format.currentData())
        processing = self.current_result.processing_spec
        segments = self.current_result.processed_segments
        start_dir = str(self.settings.value("preview/export_dir", str(self.source_path.parent)))
        jobs = []
        if len(segments) == 1 and segments[0].window_index is None:
            suggested = output_name(
                self.source_path,
                processing,
                fmt,
                extraction=self.current_result.extraction_spec,
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "导出预处理结果",
                str(Path(start_dir) / suggested),
                "EDF (*.edf)" if fmt == OutputFormat.EDF else "FIF (*.fif)",
            )
            if not path:
                return
            jobs.append((segments[0].raw, Path(path)))
            export_dir = Path(path).parent
        else:
            directory = QFileDialog.getExistingDirectory(self, "选择窗口输出目录", start_dir)
            if not directory:
                return
            export_dir = Path(directory)
            jobs = [
                (
                    segment.raw,
                    export_dir / output_name(
                        self.source_path,
                        processing,
                        fmt,
                        extraction=self.current_result.extraction_spec,
                        window_index=segment.window_index,
                    ),
                )
                for segment in segments
            ]
        existing = [path for _, path in jobs if path.exists()]
        if existing and QMessageBox.question(
            self,
            "覆盖已有文件",
            f"{len(existing)} 个目标文件已经存在。是否覆盖？",
        ) != QMessageBox.Yes:
            return
        self.settings.setValue("preview/export_dir", str(export_dir))
        self._set_busy(True)
        worker = ExportWorker(jobs, fmt)
        worker.progress.connect(self._progress_changed)
        self._start_worker(worker, self._exported, self._preview_failed)

    def _exported(self, paths):
        self.statusChanged.emit(f"已导出 {len(paths)} 个文件")


class BatchPage(ThreadedPage):
    STATUS_TEXT = {
        "ready": "○ 就绪",
        "running": "↻ 处理中",
        "done": "✓ 完成",
        "warning": "⚠ 带警告完成",
        "failed": "✕ 失败",
        "skipped": "— 已跳过",
        "cancelled": "■ 已取消",
    }

    def __init__(
        self,
        artifact_store: ArtifactSettingsStore | None = None,
        settings: QSettings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings or QSettings(
            SETTINGS_ORGANIZATION, SETTINGS_APPLICATION
        )
        self.artifact_store = artifact_store or ArtifactSettingsStore(self.settings, self)
        self.files: list[Path] = []
        self.failed_files: list[Path] = []
        self._row_by_path: dict[str, int] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        title = QLabel("批量处理")
        title.setObjectName("PageTitle")
        subtitle = QLabel("递归发现 EEG 文件，逐文件执行同一 ECMAD 降噪方案")
        subtitle.setObjectName("Muted")
        root.addWidget(title)
        root.addWidget(subtitle)

        path_panel = QFrame()
        path_panel.setObjectName("Panel")
        path_layout = QVBoxLayout(path_panel)
        self.input_edit = QLineEdit(str(self.settings.value("batch/input", "")))
        self.output_edit = QLineEdit(str(self.settings.value("batch/output", "")))
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("输入目录"))
        input_row.addWidget(self.input_edit, 1)
        browse_input = QPushButton("选择")
        browse_input.clicked.connect(self.choose_input)
        input_row.addWidget(browse_input)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("输出目录"))
        output_row.addWidget(self.output_edit, 1)
        browse_output = QPushButton("选择")
        browse_output.clicked.connect(self.choose_output)
        output_row.addWidget(browse_output)
        self.scan_button = QPushButton("扫描 EEG 文件")
        self.scan_button.clicked.connect(self.scan)
        input_row.addWidget(self.scan_button)
        path_layout.addLayout(input_row)
        path_layout.addLayout(output_row)
        root.addWidget(path_panel)

        splitter = QSplitter(Qt.Horizontal)
        table_host = QWidget()
        table_layout = QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("尚未扫描")
        self.summary.setObjectName("Metadata")
        table_layout.addWidget(self.summary)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["状态", "文件", "耗时", "输出", "消息"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        table_layout.addWidget(self.table, 1)
        self.overall_progress = QProgressBar()
        self.stage_progress = QProgressBar()
        table_layout.addWidget(self.overall_progress)
        table_layout.addWidget(self.stage_progress)
        splitter.addWidget(table_host)

        controls_host = QWidget()
        controls_host.setObjectName("ScrollContents")
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters = ParameterPanel(
            allow_selection=False, artifact_store=self.artifact_store
        )
        self.parameters.bind_settings(self.settings, "batch/parameters")
        controls_layout.addWidget(self.parameters)
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("输出格式"))
        self.output_format = _output_format_combo()
        format_row.addWidget(self.output_format, 1)
        controls_layout.addLayout(format_row)
        self.overwrite = QCheckBox("覆盖已有同名结果")
        controls_layout.addWidget(self.overwrite)
        self.start_button = QPushButton("开始批量处理")
        self.start_button.setObjectName("Primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.retry_button = QPushButton("重试失败项")
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(self.retry_failed)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.retry_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_host)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(410)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 350])
        root.addWidget(splitter, 1)

    def choose_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入目录", self.input_edit.text())
        if path:
            self.input_edit.setText(path)
            self.settings.setValue("batch/input", path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)
            self.settings.setValue("batch/output", path)

    def _set_busy(self, busy: bool):
        self.scan_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy and bool(self.files))
        self.cancel_button.setEnabled(busy)
        self.retry_button.setEnabled(not busy and bool(self.failed_files))

    def scan(self):
        root = Path(self.input_edit.text()).expanduser()
        self._set_busy(True)
        self.statusChanged.emit("正在递归扫描 EDF/FIF/SET…")
        if not self._start_worker(ScanWorker(root), self._scanned, self._batch_failed):
            self._set_busy(False)

    def _scanned(self, files):
        self.files = files
        self.failed_files = []
        self._populate_table(files)
        self.summary.setText(f"发现 {len(files)} 个可读取候选文件")
        self._set_busy(False)
        self.statusChanged.emit("扫描完成" if files else "未发现 EDF/FIF/SET 文件")

    def _populate_table(self, files: list[Path]):
        self.table.setRowCount(len(files))
        self._row_by_path.clear()
        for row, path in enumerate(files):
            self._row_by_path[str(path)] = row
            self.table.setItem(row, 0, QTableWidgetItem(self.STATUS_TEXT["ready"]))
            path_item = QTableWidgetItem(str(path))
            path_item.setToolTip(str(path))
            self.table.setItem(row, 1, path_item)
            for col in (2, 3, 4):
                self.table.setItem(row, col, QTableWidgetItem(""))

    def start_batch(self, files: list[Path] | None = None):
        chosen = files or self.files
        if not chosen:
            return
        try:
            processing = self.parameters.processing_spec()
            extraction = self.parameters.extraction_spec()
            processing.validate()
            extraction.validate()
            input_root = Path(self.input_edit.text())
            output_root = Path(self.output_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "参数需要调整", str(exc))
            return
        self.settings.setValue("batch/input", str(input_root))
        self.settings.setValue("batch/output", str(output_root))
        self.failed_files = []
        self._set_busy(True)
        self.overall_progress.setRange(0, len(chosen))
        self.overall_progress.setValue(0)
        worker = BatchWorker(
            chosen,
            input_root,
            output_root,
            processing,
            extraction,
            self.artifact_store.snapshot(),
            OutputFormat(self.output_format.currentData()),
            self.overwrite.isChecked(),
        )
        worker.itemStarted.connect(self._item_started)
        worker.itemFinished.connect(self._item_finished)
        worker.progress.connect(self._stage_changed)
        if not self._start_worker(worker, self._batch_finished, self._batch_failed):
            self._set_busy(False)

    def _item_started(self, index: int, total: int, path: str):
        self.overall_progress.setMaximum(max(1, total))
        self.overall_progress.setValue(index)
        if path in self._row_by_path:
            self.table.item(self._row_by_path[path], 0).setText(self.STATUS_TEXT["running"])
        self.statusChanged.emit(f"正在处理 {index + 1}/{total} · {Path(path).name}")

    def _stage_changed(self, current: int, total: int, label: str):
        self.stage_progress.setRange(0, max(1, total))
        self.stage_progress.setValue(current)
        self.stage_progress.setFormat(f"{label} %v/%m")

    def _item_finished(self, result):
        row = self._row_by_path.get(str(result.source))
        if row is None:
            return
        self.table.item(row, 0).setText(self.STATUS_TEXT.get(result.status, result.status))
        self.table.item(row, 2).setText(f"{result.elapsed_sec:.2f} s")
        self.table.item(row, 3).setText(" | ".join(str(p) for p in result.outputs))
        message = result.error or "；".join(result.warnings)
        self.table.item(row, 4).setText(message)
        if result.status == "failed":
            self.failed_files.append(result.source)

    def _batch_finished(self, results):
        self.overall_progress.setValue(len(results))
        self._set_busy(False)
        counts = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        self.summary.setText(" · ".join(f"{key}: {value}" for key, value in counts.items()))
        self.statusChanged.emit("批量任务已结束")

    def _batch_failed(self, message: str):
        self._set_busy(False)
        self._task_failed(message)

    def _task_cancelled(self, message: str):
        self._set_busy(False)
        self.summary.setText("批量任务已取消；已完成项与清单保持有效")
        super()._task_cancelled(message)

    def retry_failed(self):
        self.start_batch(list(self.failed_files))
