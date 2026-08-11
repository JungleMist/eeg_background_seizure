from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QSettings, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionSpinBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eeg_bg.application.models import (
    ArtifactSettings,
    ExtractionMode,
    ExtractionSpec,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
)

from .branding import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION
from .channel_groups import ChannelGroupEditorDialog, ChannelGroupPresetStore
from .theme import COLORS


class ArtifactSettingsStore(QObject):
    changed = Signal(object)

    ENABLED_KEY = "global/artifact_threshold_enabled"
    THRESHOLD_KEY = "global/artifact_threshold_uv"

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        stored_enabled = settings.value(self.ENABLED_KEY, True)
        self._enabled = (
            stored_enabled.lower() in {"1", "true", "yes", "on"}
            if isinstance(stored_enabled, str)
            else bool(stored_enabled)
        )
        self._threshold_uv = float(settings.value(self.THRESHOLD_KEY, 200.0))
        self.snapshot().validate()

    def snapshot(self) -> ArtifactSettings:
        return ArtifactSettings(self._enabled, self._threshold_uv)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._settings.setValue(self.ENABLED_KEY, enabled)
        self.changed.emit(self.snapshot())

    def set_threshold_uv(self, threshold_uv: float) -> None:
        threshold_uv = float(threshold_uv)
        ArtifactSettings(self._enabled, threshold_uv).validate()
        if threshold_uv == self._threshold_uv:
            return
        self._threshold_uv = threshold_uv
        self._settings.setValue(self.THRESHOLD_KEY, threshold_uv)
        self.changed.emit(self.snapshot())


class _LargeStepperMixin:
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(
            QColor(COLORS["text"] if self.isEnabled() else COLORS["muted"]),
            2.0,
        )
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        for subcontrol, direction in (
            (QStyle.SC_SpinBoxUp, 1.0),
            (QStyle.SC_SpinBoxDown, -1.0),
        ):
            target = self.style().subControlRect(
                QStyle.CC_SpinBox, option, subcontrol, self
            )
            center = target.center()
            x = float(center.x())
            y = float(center.y())
            painter.drawLine(
                QPointF(x - 5.0, y + 2.5 * direction),
                QPointF(x, y - 2.5 * direction),
            )
            painter.drawLine(
                QPointF(x, y - 2.5 * direction),
                QPointF(x + 5.0, y + 2.5 * direction),
            )


class LargeStepDoubleSpinBox(_LargeStepperMixin, QDoubleSpinBox):
    pass


class LargeStepSpinBox(_LargeStepperMixin, QSpinBox):
    pass


class ParameterPanel(QWidget):
    parametersChanged = Signal()
    currentWindowRequested = Signal()

    def __init__(
        self,
        *,
        allow_selection: bool,
        artifact_store: ArtifactSettingsStore | None = None,
        channel_group_presets: ChannelGroupPresetStore | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.allow_selection = allow_selection
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.artifact_store = artifact_store or ArtifactSettingsStore(
            QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION), self
        )
        self.channel_group_presets = channel_group_presets
        if allow_selection and self.channel_group_presets is None:
            self.channel_group_presets = ChannelGroupPresetStore(
                QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION), self
            )
        global_group = QGroupBox("全局设置")
        self.global_form = QFormLayout(global_group)
        self.artifact_enabled = QCheckBox("启用伪迹阈值")
        self.artifact_uv = self._double(1.0, 10000.0, 10.0, 200.0, 1)
        self.global_form.addRow(self.artifact_enabled)
        self.global_form.addRow("阈值 (µV)", self.artifact_uv)
        self.artifact_note = QLabel("仅用于红色标记和批量警告，不改变任何处理结果。")
        self.artifact_note.setObjectName("Muted")
        self.artifact_note.setWordWrap(True)
        self.global_form.addRow(self.artifact_note)
        layout.addWidget(global_group)

        extraction_group = QGroupBox("提取范围")
        self.extraction_form = QFormLayout(extraction_group)
        self.extraction_mode = QComboBox()
        if allow_selection:
            self.extraction_mode.addItem("交互选区", ExtractionMode.SELECTION)
        self.extraction_mode.addItem("固定窗口", ExtractionMode.FIXED_WINDOWS)
        self.extraction_mode.addItem("完整长序列", ExtractionMode.CONTINUOUS)
        self.start_sec = self._double(0.0, 864000.0, 0.1, 0.0, 2)
        self.stop_sec = self._double(0.0, 864000.0, 0.1, 20.0, 2)
        self.current_window_button = QPushButton("当前窗口")
        self.current_window_button.setToolTip(
            "使用波形图当前可见的时间范围作为交互选区"
        )
        self.current_window_button.setAccessibleName("使用当前可见窗口作为选区")
        self.current_window_button.clicked.connect(self.currentWindowRequested)
        self._current_window_available = False
        self.window_sec = self._double(4.0, 600.0, 1.0, 20.0, 1)
        self.extraction_form.addRow("提取方式", self.extraction_mode)
        if allow_selection:
            self.extraction_form.addRow("起点 (s)", self.start_sec)
            self.extraction_form.addRow("终点 (s)", self.stop_sec)
            self.extraction_form.addRow("快捷范围", self.current_window_button)
        self.extraction_form.addRow("固定分段长度 (s)", self.window_sec)
        layout.addWidget(extraction_group)

        base_group = QGroupBox("基础预处理")
        self.base_form = QFormLayout(base_group)
        self.low_hz = self._double(0.1, 500.0, 0.1, 0.5, 1)
        self.high_hz = self._double(0.2, 1000.0, 1.0, 40.0, 1)
        self.sfreq = self._double(10.0, 4096.0, 25.0, 125.0, 1)
        self.analysis_window = self._double(4.0, 600.0, 1.0, 20.0, 1)
        self.base_form.addRow("带通低限 (Hz)", self.low_hz)
        self.base_form.addRow("带通高限 (Hz)", self.high_hz)
        self.base_form.addRow("目标采样率 (Hz)", self.sfreq)
        self.base_form.addRow("算法分析窗长 (s)", self.analysis_window)
        layout.addWidget(base_group)

        method_group = QGroupBox("处理方法")
        self.method_form = QFormLayout(method_group)
        self.method = QComboBox()
        self.method.addItem("仅基础处理", ProcessingMethod.BASIC)
        self.method.addItem("ICA", ProcessingMethod.ICA)
        self.method.addItem("ECMAD (Wiener)", ProcessingMethod.WIENER)
        self.ica_components = LargeStepSpinBox()
        self.ica_components.setRange(0, 19)
        self.ica_components.setSpecialValueText("自动")
        self._configure_spinbox(self.ica_components)
        self.ica_threshold = self._double(0.0, 1.0, 0.01, 0.80, 2)
        self.wiener_mode = QComboBox()
        self.wiener_mode.addItem("frequency", WienerMode.FREQUENCY)
        self.wiener_mode.addItem("phasegated", WienerMode.PHASEGATED)
        self.wiener_mode.addItem("zerophase", WienerMode.ZEROPHASE)
        self.coherence = self._double(0.0, 1.0, 0.01, 0.15, 2)
        self.coherent_gate_enabled = QCheckBox("启用 coherent 功率门控")
        self.coherent_gate_enabled.setChecked(True)
        self.coherent_gate_threshold_uv = self._double(
            0.1, 100000.0, 10.0, 100.0, 1
        )
        self.gate = self._double(0.0, 3.14, 0.01, 0.39, 2)
        self.gate.setToolTip("单位为弧度；3.14 会映射为精确 π")
        self.protected_band_enabled = QCheckBox("启用相干成分频带保护")
        self.protected_band_enabled.setChecked(True)
        self.protected_low_hz = self._double(0.0, 1000.0, 0.5, 5.0, 1)
        self.protected_high_hz = self._double(0.0, 1000.0, 0.5, 20.0, 1)
        self.channel_groups_button = QPushButton("加载 EEG 后编辑")
        self.channel_groups_button.setAccessibleName("编辑 ECMAD 导联组")
        self.channel_groups_button.setToolTip(
            "使用当前 EEG 存在的导联编辑 ECMAD 传导路径"
        )
        self.channel_groups_button.clicked.connect(self.open_channel_group_editor)
        self._available_channels: tuple[str, ...] = ()
        self._channel_groups: tuple[tuple[str, ...], ...] = ()
        self.method_form.addRow("处理方法", self.method)
        self.method_form.addRow("ICA 组件", self.ica_components)
        self.method_form.addRow("ICA 相关阈值", self.ica_threshold)
        self.method_form.addRow("ECMAD 模式", self.wiener_mode)
        self.method_form.addRow("coherence", self.coherence)
        self.method_form.addRow(self.coherent_gate_enabled)
        self.method_form.addRow(
            "coherent 阈值 (µV)", self.coherent_gate_threshold_uv
        )
        self.method_form.addRow("gate (rad)", self.gate)
        self.method_form.addRow(self.protected_band_enabled)
        self.method_form.addRow("保护低限 (Hz)", self.protected_low_hz)
        self.method_form.addRow("保护高限 (Hz)", self.protected_high_hz)
        if allow_selection:
            self.method_form.addRow("导联组", self.channel_groups_button)
        layout.addWidget(method_group)

        self.hint = QLabel()
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        layout.addStretch(1)

        self.extraction_mode.currentIndexChanged.connect(self._sync_visibility)
        self.method.currentIndexChanged.connect(self._sync_visibility)
        self.wiener_mode.currentIndexChanged.connect(self._sync_visibility)
        self.coherent_gate_enabled.toggled.connect(self._sync_visibility)
        self.coherent_gate_enabled.toggled.connect(self.parametersChanged)
        self.protected_band_enabled.toggled.connect(self._sync_visibility)
        self.protected_band_enabled.toggled.connect(self.parametersChanged)
        self.artifact_enabled.toggled.connect(self.artifact_store.set_enabled)
        self.artifact_uv.valueChanged.connect(self.artifact_store.set_threshold_uv)
        self.artifact_store.changed.connect(self._artifact_settings_changed)
        controls = (
            self.findChildren(QComboBox)
            + self.findChildren(QSpinBox)
            + self.findChildren(QDoubleSpinBox)
        )
        for control in controls:
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self.parametersChanged)
            else:
                control.valueChanged.connect(self.parametersChanged)
        self._artifact_settings_changed(self.artifact_store.snapshot())
        self._sync_visibility()

    def bind_settings(self, settings: QSettings, prefix: str) -> None:
        numeric = {
            "start_sec": self.start_sec,
            "stop_sec": self.stop_sec,
            "window_sec": self.window_sec,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "sfreq": self.sfreq,
            "analysis_window": self.analysis_window,
            "ica_components": self.ica_components,
            "ica_threshold": self.ica_threshold,
            "coherence": self.coherence,
            "coherent_gate_threshold_uv": self.coherent_gate_threshold_uv,
            "gate": self.gate,
            "protected_low_hz": self.protected_low_hz,
            "protected_high_hz": self.protected_high_hz,
        }
        combos = {
            "extraction_mode": self.extraction_mode,
            "method": self.method,
            "wiener_mode": self.wiener_mode,
        }
        for key, widget in numeric.items():
            stored = settings.value(f"{prefix}/{key}")
            if stored is not None:
                widget.setValue(float(stored) if isinstance(widget, QDoubleSpinBox) else int(stored))
        for key, combo in combos.items():
            stored = settings.value(f"{prefix}/{key}")
            if stored is None:
                continue
            for index in range(combo.count()):
                value = combo.itemData(index)
                if getattr(value, "value", value) == stored:
                    combo.setCurrentIndex(index)
                    break
        stored_protected = settings.value(f"{prefix}/protected_band_enabled")
        if stored_protected is not None:
            enabled = (
                stored_protected.lower() in {"1", "true", "yes", "on"}
                if isinstance(stored_protected, str)
                else bool(stored_protected)
            )
            self.protected_band_enabled.setChecked(enabled)
        stored_coherent_gate = settings.value(
            f"{prefix}/coherent_gate_enabled"
        )
        if stored_coherent_gate is not None:
            enabled = (
                stored_coherent_gate.lower() in {"1", "true", "yes", "on"}
                if isinstance(stored_coherent_gate, str)
                else bool(stored_coherent_gate)
            )
            self.coherent_gate_enabled.setChecked(enabled)
        self._sync_visibility()

        def save():
            for key, widget in numeric.items():
                settings.setValue(f"{prefix}/{key}", widget.value())
            for key, combo in combos.items():
                value = combo.currentData()
                settings.setValue(f"{prefix}/{key}", getattr(value, "value", value))
            settings.setValue(
                f"{prefix}/protected_band_enabled",
                self.protected_band_enabled.isChecked(),
            )
            settings.setValue(
                f"{prefix}/coherent_gate_enabled",
                self.coherent_gate_enabled.isChecked(),
            )

        self.parametersChanged.connect(save)

    @staticmethod
    def _double(low, high, step, value, decimals):
        widget = LargeStepDoubleSpinBox()
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setValue(value)
        ParameterPanel._configure_spinbox(widget)
        return widget

    @staticmethod
    def _configure_spinbox(widget: QAbstractSpinBox) -> None:
        # The platform-default steppers are only a few pixels wide. The theme
        # gives this reserved area two large, clearly separated arrow targets.
        widget.setMinimumHeight(46)
        widget.setMinimumWidth(132)
        widget.setAccelerated(True)

    @staticmethod
    def _set_row_visible(form: QFormLayout, widget: QWidget, visible: bool) -> None:
        form.setRowVisible(widget, visible)
        widget.setEnabled(visible)

    def _sync_visibility(self):
        extraction = ExtractionMode(self.extraction_mode.currentData())
        method = ProcessingMethod(self.method.currentData())
        selection = extraction == ExtractionMode.SELECTION
        fixed = extraction == ExtractionMode.FIXED_WINDOWS
        is_ica = method == ProcessingMethod.ICA
        is_wiener = method == ProcessingMethod.WIENER

        if self.allow_selection:
            self._set_row_visible(self.extraction_form, self.start_sec, selection)
            self._set_row_visible(self.extraction_form, self.stop_sec, selection)
            self._set_row_visible(
                self.extraction_form, self.current_window_button, selection
            )
            self.current_window_button.setEnabled(
                selection and self._current_window_available
            )
        self._set_row_visible(self.extraction_form, self.window_sec, fixed)
        self._set_row_visible(
            self.base_form,
            self.analysis_window,
            not fixed and (is_ica or is_wiener),
        )

        self._set_row_visible(self.method_form, self.ica_components, is_ica)
        self._set_row_visible(self.method_form, self.ica_threshold, is_ica)
        self._set_row_visible(self.method_form, self.wiener_mode, is_wiener)
        self._set_row_visible(self.method_form, self.coherence, is_wiener)
        self._set_row_visible(
            self.method_form, self.coherent_gate_enabled, is_wiener
        )
        self._set_row_visible(
            self.method_form,
            self.coherent_gate_threshold_uv,
            is_wiener and self.coherent_gate_enabled.isChecked(),
        )
        self._set_row_visible(
            self.method_form, self.protected_band_enabled, is_wiener
        )
        protected_enabled = (
            is_wiener and self.protected_band_enabled.isChecked()
        )
        self._set_row_visible(
            self.method_form, self.protected_low_hz, protected_enabled
        )
        self._set_row_visible(
            self.method_form, self.protected_high_hz, protected_enabled
        )
        if self.allow_selection:
            self._set_row_visible(
                self.method_form, self.channel_groups_button, is_wiener
            )
            self.channel_groups_button.setEnabled(
                is_wiener and len(self._available_channels) >= 2
            )
        self._set_row_visible(
            self.method_form,
            self.gate,
            is_wiener
            and WienerMode(self.wiener_mode.currentData()) != WienerMode.FREQUENCY
        )

        extraction_hint = {
            ExtractionMode.SELECTION: "仅处理起点到终点之间的 EEG。",
            ExtractionMode.FIXED_WINDOWS: (
                "按固定分段长度切分并处理全部完整分段；不足一段的尾部舍弃。"
            ),
            ExtractionMode.CONTINUOUS: "处理并导出完整长序列。",
        }[extraction]
        method_hint = {
            ProcessingMethod.BASIC: "基础处理不需要算法分析窗。",
            ProcessingMethod.ICA: (
                "ICA 直接基于全部固定分段拟合。"
                if fixed
                else "算法分析窗用于 ICA 拟合分块。"
            ),
            ProcessingMethod.WIENER: (
                "ECMAD 通过 Wiener 算法逐固定分段处理。"
                if fixed
                else "算法分析窗用于 50% 重叠的 ECMAD Wiener 处理。"
            ),
        }[method]
        self.hint.setText(f"{extraction_hint} {method_hint}")

    def _artifact_settings_changed(self, settings: ArtifactSettings) -> None:
        with QSignalBlocker(self.artifact_enabled):
            self.artifact_enabled.setChecked(settings.enabled)
        with QSignalBlocker(self.artifact_uv):
            self.artifact_uv.setValue(settings.threshold_uv)
        self._set_row_visible(self.global_form, self.artifact_uv, settings.enabled)

    def artifact_settings(self) -> ArtifactSettings:
        return self.artifact_store.snapshot()

    def set_selection(self, start: float, stop: float):
        if not self.allow_selection:
            return
        self.start_sec.blockSignals(True)
        self.stop_sec.blockSignals(True)
        self.start_sec.setValue(max(0.0, start))
        self.stop_sec.setValue(max(start, stop))
        self.start_sec.blockSignals(False)
        self.stop_sec.blockSignals(False)

    def set_current_window_available(self, available: bool) -> None:
        self._current_window_available = bool(available)
        self._sync_visibility()

    def set_channel_context(
        self,
        channels: list[str] | tuple[str, ...],
        default_groups: list[list[str]] | tuple[tuple[str, ...], ...],
    ) -> None:
        self._available_channels = tuple(dict.fromkeys(channels))
        available = set(self._available_channels)
        self._channel_groups = tuple(
            tuple(group)
            for group in default_groups
            if len(group) >= 2 and all(channel in available for channel in group)
        )
        self._update_channel_group_button()
        self._sync_visibility()

    def open_channel_group_editor(self) -> None:
        if len(self._available_channels) < 2:
            return
        dialog = ChannelGroupEditorDialog(
            self._available_channels,
            self._channel_groups,
            preset_store=self.channel_group_presets,
            parent=self,
        )
        if dialog.exec():
            self._channel_groups = dialog.groups()
            self._update_channel_group_button()
            self.parametersChanged.emit()

    def _update_channel_group_button(self) -> None:
        if not self._available_channels:
            self.channel_groups_button.setText("加载 EEG 后编辑")
            return
        count = len(self._channel_groups)
        text = f"编辑（{count} 组）" if count else "编辑（尚未配置）"
        self.channel_groups_button.setText(text)

    def processing_spec(self) -> ProcessingSpec:
        components = self.ica_components.value()
        method = ProcessingMethod(self.method.currentData())
        return ProcessingSpec(
            method=method,
            bandpass_low_hz=self.low_hz.value(),
            bandpass_high_hz=self.high_hz.value(),
            target_sfreq=self.sfreq.value(),
            analysis_window_sec=self.analysis_window.value(),
            ica_n_components=components or None,
            ica_artifact_corr_threshold=self.ica_threshold.value(),
            wiener_mode=WienerMode(self.wiener_mode.currentData()),
            coherence_threshold=self.coherence.value(),
            coherent_gate_enabled=self.coherent_gate_enabled.isChecked(),
            coherent_gate_threshold_uv=self.coherent_gate_threshold_uv.value(),
            phase_gate_threshold_rad=self.gate.value(),
            protected_band_hz=(
                (
                    self.protected_low_hz.value(),
                    self.protected_high_hz.value(),
                )
                if self.protected_band_enabled.isChecked()
                else None
            ),
            channel_groups=(
                self._channel_groups
                if self.allow_selection and method == ProcessingMethod.WIENER
                else None
            ),
        )

    def extraction_spec(self) -> ExtractionSpec:
        stop = self.stop_sec.value() if self.allow_selection else None
        return ExtractionSpec(
            mode=ExtractionMode(self.extraction_mode.currentData()),
            start_sec=self.start_sec.value() if self.allow_selection else 0.0,
            stop_sec=stop,
            window_sec=self.window_sec.value(),
        )
