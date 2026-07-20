from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eeg_bg.application.models import (
    ExtractionMode,
    ExtractionSpec,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
)


class ParameterPanel(QWidget):
    parametersChanged = Signal()

    def __init__(self, *, allow_selection: bool, parent=None):
        super().__init__(parent)
        self.allow_selection = allow_selection
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        extraction_group = QGroupBox("提取范围")
        extraction_form = QFormLayout(extraction_group)
        self.extraction_mode = QComboBox()
        if allow_selection:
            self.extraction_mode.addItem("交互选区", ExtractionMode.SELECTION)
        self.extraction_mode.addItem("固定窗口", ExtractionMode.FIXED_WINDOWS)
        self.extraction_mode.addItem("完整长序列", ExtractionMode.CONTINUOUS)
        self.start_sec = self._double(0.0, 864000.0, 0.1, 0.0, 2)
        self.stop_sec = self._double(0.0, 864000.0, 0.1, 20.0, 2)
        self.window_sec = self._double(4.0, 600.0, 1.0, 20.0, 1)
        self.artifact_uv = self._double(1.0, 10000.0, 10.0, 200.0, 1)
        extraction_form.addRow("提取方式", self.extraction_mode)
        if allow_selection:
            extraction_form.addRow("起点 (s)", self.start_sec)
            extraction_form.addRow("终点 (s)", self.stop_sec)
        extraction_form.addRow("窗口长度 (s)", self.window_sec)
        extraction_form.addRow("伪迹阈值 (µV)", self.artifact_uv)
        layout.addWidget(extraction_group)

        base_group = QGroupBox("基础预处理")
        base_form = QFormLayout(base_group)
        self.low_hz = self._double(0.1, 500.0, 0.1, 0.5, 1)
        self.high_hz = self._double(0.2, 1000.0, 1.0, 40.0, 1)
        self.sfreq = self._double(10.0, 4096.0, 25.0, 125.0, 1)
        self.analysis_window = self._double(4.0, 600.0, 1.0, 20.0, 1)
        base_form.addRow("带通低限 (Hz)", self.low_hz)
        base_form.addRow("带通高限 (Hz)", self.high_hz)
        base_form.addRow("目标采样率 (Hz)", self.sfreq)
        base_form.addRow("分析窗长 (s)", self.analysis_window)
        layout.addWidget(base_group)

        method_group = QGroupBox("处理方案")
        method_form = QFormLayout(method_group)
        self.method = QComboBox()
        self.method.addItem("仅基础处理", ProcessingMethod.BASIC)
        self.method.addItem("ICA", ProcessingMethod.ICA)
        self.method.addItem("Wiener", ProcessingMethod.WIENER)
        self.ica_components = QSpinBox()
        self.ica_components.setRange(0, 19)
        self.ica_components.setSpecialValueText("自动")
        self.ica_threshold = self._double(0.0, 1.0, 0.01, 0.80, 2)
        self.wiener_mode = QComboBox()
        self.wiener_mode.addItem("frequency", WienerMode.FREQUENCY)
        self.wiener_mode.addItem("phasegated", WienerMode.PHASEGATED)
        self.wiener_mode.addItem("zerophase", WienerMode.ZEROPHASE)
        self.coherence = self._double(0.0, 1.0, 0.01, 0.15, 2)
        self.gate = self._double(0.0, 3.14, 0.01, 0.39, 2)
        self.gate.setToolTip("单位为弧度；3.14 会映射为精确 π")
        method_form.addRow("方法", self.method)
        method_form.addRow("ICA 组件", self.ica_components)
        method_form.addRow("ICA 相关阈值", self.ica_threshold)
        method_form.addRow("Wiener 模式", self.wiener_mode)
        method_form.addRow("coherence", self.coherence)
        method_form.addRow("gate (rad)", self.gate)
        layout.addWidget(method_group)

        hint = QLabel("固定窗口会丢弃超过伪迹阈值的窗口；完整长序列保持原始时长。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

        self.extraction_mode.currentIndexChanged.connect(self._sync_enabled)
        self.method.currentIndexChanged.connect(self._sync_enabled)
        self.wiener_mode.currentIndexChanged.connect(self._sync_enabled)
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
        self._sync_enabled()

    def bind_settings(self, settings: QSettings, prefix: str) -> None:
        numeric = {
            "start_sec": self.start_sec,
            "stop_sec": self.stop_sec,
            "window_sec": self.window_sec,
            "artifact_uv": self.artifact_uv,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "sfreq": self.sfreq,
            "analysis_window": self.analysis_window,
            "ica_components": self.ica_components,
            "ica_threshold": self.ica_threshold,
            "coherence": self.coherence,
            "gate": self.gate,
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
        self._sync_enabled()

        def save():
            for key, widget in numeric.items():
                settings.setValue(f"{prefix}/{key}", widget.value())
            for key, combo in combos.items():
                value = combo.currentData()
                settings.setValue(f"{prefix}/{key}", getattr(value, "value", value))

        self.parametersChanged.connect(save)

    @staticmethod
    def _double(low, high, step, value, decimals):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setValue(value)
        return widget

    def _sync_enabled(self):
        extraction = self.extraction_mode.currentData()
        if self.allow_selection:
            selection = extraction == ExtractionMode.SELECTION
            self.start_sec.setEnabled(selection)
            self.stop_sec.setEnabled(selection)
        self.window_sec.setEnabled(extraction != ExtractionMode.CONTINUOUS)

        method = self.method.currentData()
        is_ica = method == ProcessingMethod.ICA
        is_wiener = method == ProcessingMethod.WIENER
        self.ica_components.setEnabled(is_ica)
        self.ica_threshold.setEnabled(is_ica)
        self.wiener_mode.setEnabled(is_wiener)
        self.coherence.setEnabled(is_wiener)
        self.gate.setEnabled(
            is_wiener and self.wiener_mode.currentData() != WienerMode.FREQUENCY
        )

    def set_selection(self, start: float, stop: float):
        if not self.allow_selection:
            return
        self.start_sec.blockSignals(True)
        self.stop_sec.blockSignals(True)
        self.start_sec.setValue(max(0.0, start))
        self.stop_sec.setValue(max(start, stop))
        self.start_sec.blockSignals(False)
        self.stop_sec.blockSignals(False)

    def processing_spec(self) -> ProcessingSpec:
        components = self.ica_components.value()
        return ProcessingSpec(
            method=self.method.currentData(),
            bandpass_low_hz=self.low_hz.value(),
            bandpass_high_hz=self.high_hz.value(),
            target_sfreq=self.sfreq.value(),
            analysis_window_sec=self.analysis_window.value(),
            ica_n_components=components or None,
            ica_artifact_corr_threshold=self.ica_threshold.value(),
            wiener_mode=self.wiener_mode.currentData(),
            coherence_threshold=self.coherence.value(),
            phase_gate_threshold_rad=self.gate.value(),
        )

    def extraction_spec(self) -> ExtractionSpec:
        stop = self.stop_sec.value() if self.allow_selection else None
        return ExtractionSpec(
            mode=self.extraction_mode.currentData(),
            start_sec=self.start_sec.value() if self.allow_selection else 0.0,
            stop_sec=stop,
            window_sec=self.window_sec.value(),
            artifact_threshold_uv=self.artifact_uv.value(),
        )
