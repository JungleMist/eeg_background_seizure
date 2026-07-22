import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox

from eeg_bg.application.models import (
    ArtifactSettings,
    ExtractionMode,
    ProcessingMethod,
    WienerMode,
)
from eeg_bg.gui.branding import APPLICATION_NAME, PRODUCT_NAME
from eeg_bg.gui.main_window import MainWindow
from eeg_bg.gui.pages import BatchPage, PreviewPage
from eeg_bg.gui.parameters import ArtifactSettingsStore, ParameterPanel
from eeg_bg.gui.theme import COLORS, stylesheet
from eeg_bg.gui.waveform import WaveformView


def test_main_window_has_two_workspaces(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert APPLICATION_NAME in window.windowTitle()
    assert PRODUCT_NAME in window.windowTitle()
    assert window.stack.count() == 2
    qtbot.mouseClick(window.batch_nav, Qt.LeftButton)
    assert window.stack.currentIndex() == 1


def test_preview_advertises_eeglab_set_input(qtbot):
    page = PreviewPage()
    qtbot.addWidget(page)
    assert "EEGLAB SET" in page.metadata.text()
    assert "FDT" in page.metadata.text()


def test_parameter_panel_enables_gate_only_for_gated_modes(qtbot):
    panel = ParameterPanel(allow_selection=True)
    qtbot.addWidget(panel)
    assert "ECMAD" in panel.method.itemText(
        panel.method.findData(ProcessingMethod.WIENER)
    )
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.WIENER))
    panel.wiener_mode.setCurrentIndex(panel.wiener_mode.findData(WienerMode.FREQUENCY))
    assert not panel.gate.isEnabled()
    panel.wiener_mode.setCurrentIndex(panel.wiener_mode.findData(WienerMode.PHASEGATED))
    assert panel.gate.isEnabled()


def test_parameter_panel_shows_only_effective_parameters(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "panel.ini"), QSettings.IniFormat)
    panel = ParameterPanel(
        allow_selection=True,
        artifact_store=ArtifactSettingsStore(settings),
    )
    qtbot.addWidget(panel)

    # Interactive selection + basic processing.
    assert panel.extraction_form.isRowVisible(panel.start_sec)
    assert panel.extraction_form.isRowVisible(panel.stop_sec)
    assert not panel.extraction_form.isRowVisible(panel.window_sec)
    assert panel.global_form.isRowVisible(panel.artifact_uv)
    assert not panel.base_form.isRowVisible(panel.analysis_window)
    assert not panel.method_form.isRowVisible(panel.ica_components)
    assert not panel.method_form.isRowVisible(panel.wiener_mode)

    panel.window_sec.setValue(42.0)
    panel.extraction_mode.setCurrentIndex(
        panel.extraction_mode.findData(ExtractionMode.FIXED_WINDOWS)
    )
    assert not panel.extraction_form.isRowVisible(panel.start_sec)
    assert not panel.extraction_form.isRowVisible(panel.stop_sec)
    assert panel.extraction_form.isRowVisible(panel.window_sec)
    assert not panel.base_form.isRowVisible(panel.analysis_window)

    panel.extraction_mode.setCurrentIndex(
        panel.extraction_mode.findData(ExtractionMode.CONTINUOUS)
    )
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.ICA))
    assert not panel.extraction_form.isRowVisible(panel.window_sec)
    assert panel.base_form.isRowVisible(panel.analysis_window)
    assert panel.method_form.isRowVisible(panel.ica_components)
    assert panel.method_form.isRowVisible(panel.ica_threshold)
    assert not panel.method_form.isRowVisible(panel.wiener_mode)

    panel.extraction_mode.setCurrentIndex(
        panel.extraction_mode.findData(ExtractionMode.SELECTION)
    )
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.WIENER))
    assert panel.base_form.isRowVisible(panel.analysis_window)
    assert panel.method_form.isRowVisible(panel.wiener_mode)
    assert panel.method_form.isRowVisible(panel.coherence)
    assert not panel.method_form.isRowVisible(panel.gate)

    panel.wiener_mode.setCurrentIndex(
        panel.wiener_mode.findData(WienerMode.PHASEGATED)
    )
    assert panel.method_form.isRowVisible(panel.gate)

    panel.extraction_mode.setCurrentIndex(
        panel.extraction_mode.findData(ExtractionMode.FIXED_WINDOWS)
    )
    assert panel.window_sec.value() == 42.0
    assert not panel.base_form.isRowVisible(panel.analysis_window)


def test_parameter_panel_returns_typed_specs_and_large_step_targets(qtbot):
    panel = ParameterPanel(allow_selection=True)
    panel.setStyleSheet(stylesheet())
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)

    assert panel.processing_spec().method is ProcessingMethod.BASIC
    assert panel.processing_spec().wiener_mode is WienerMode.FREQUENCY
    assert panel.extraction_spec().mode is ExtractionMode.SELECTION

    option = QStyleOptionSpinBox()
    panel.low_hz.initStyleOption(option)
    up_target = panel.low_hz.style().subControlRect(
        QStyle.CC_SpinBox,
        option,
        QStyle.SC_SpinBoxUp,
        panel.low_hz,
    )
    assert up_target.width() >= 36
    assert up_target.height() >= 20
    initial = panel.low_hz.value()
    qtbot.mouseClick(panel.low_hz, Qt.LeftButton, pos=up_target.center())
    assert panel.low_hz.value() > initial

    down_target = panel.low_hz.style().subControlRect(
        QStyle.CC_SpinBox,
        option,
        QStyle.SC_SpinBoxDown,
        panel.low_hz,
    )
    qtbot.mouseClick(panel.low_hz, Qt.LeftButton, pos=down_target.center())
    assert panel.low_hz.value() == initial


def test_global_artifact_settings_sync_persist_and_hide_threshold(qtbot, tmp_path):
    settings_path = tmp_path / "studio.ini"
    settings = QSettings(str(settings_path), QSettings.IniFormat)
    store = ArtifactSettingsStore(settings)
    preview = ParameterPanel(allow_selection=True, artifact_store=store)
    batch = ParameterPanel(allow_selection=False, artifact_store=store)
    qtbot.addWidget(preview)
    qtbot.addWidget(batch)

    preview.artifact_uv.setValue(345.0)
    assert batch.artifact_uv.value() == 345.0
    batch.artifact_enabled.setChecked(False)
    assert not preview.artifact_enabled.isChecked()
    assert not preview.global_form.isRowVisible(preview.artifact_uv)
    assert not batch.global_form.isRowVisible(batch.artifact_uv)

    settings.sync()
    restored = ArtifactSettingsStore(
        QSettings(str(settings_path), QSettings.IniFormat)
    ).snapshot()
    assert restored == ArtifactSettings(enabled=False, threshold_uv=345.0)


def test_waveform_region_updates_selection(qtbot):
    raw = mne.io.RawArray(
        np.zeros((2, 1000)),
        mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    with qtbot.waitSignal(view.selectionChanged, timeout=1000) as blocker:
        view.set_original(raw)
        view.region.setRegion((2.0, 6.0))
        view.region.sigRegionChangeFinished.emit(view.region)
    assert blocker.args == [2.0, 6.0]


def test_channel_menu_can_clear_all_and_stays_open_for_multiple_changes(qtbot):
    raw = mne.io.RawArray(
        np.zeros((3, 100)),
        mne.create_info(["FP1", "FP2", "F3"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_original(raw)

    menu_actions = {action.text(): action for action in view.channel_menu.actions()}
    assert "显示全部" in menu_actions
    assert "取消显示全部" in menu_actions
    menu_actions["取消显示全部"].trigger()
    assert view._visible_channels == []
    assert view.channel_button.text() == "通道 0/3"

    view.channel_menu.popup(QPoint(20, 20))
    qtbot.waitUntil(view.channel_menu.isVisible)
    fp1 = menu_actions["FP1"]
    qtbot.mouseClick(
        view.channel_menu,
        Qt.LeftButton,
        pos=view.channel_menu.actionGeometry(fp1).center(),
    )
    assert fp1.isChecked()
    assert view.channel_menu.isVisible()

    fp2 = menu_actions["FP2"]
    qtbot.mouseClick(
        view.channel_menu,
        Qt.LeftButton,
        pos=view.channel_menu.actionGeometry(fp2).center(),
    )
    assert fp2.isChecked()
    assert view.channel_menu.isVisible()
    assert view._visible_channels == ["FP1", "FP2"]
    view.channel_menu.close()


def test_waveform_marks_raw_and_processed_thresholds_independently(qtbot):
    raw_data = np.zeros((2, 20))
    raw_data[0, 2:5] = 250e-6
    processed_data = np.zeros((2, 20))
    processed_data[1, 8:12] = -300e-6
    info = mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    processed = mne.io.RawArray(processed_data, info, verbose=False)
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_artifact_settings(ArtifactSettings(enabled=True, threshold_uv=200.0))
    view.set_original(raw)
    view.set_processed(processed, 0.0)

    def red_indices(plot):
        red = []
        for item in plot.listDataItems():
            pen = item.opts.get("pen")
            if pen is not None and pen.color().name().upper() == COLORS["danger"].upper():
                _, y = item.getData()
                red.append(np.flatnonzero(np.isfinite(y)).tolist())
        return red

    assert red_indices(view.raw_plot) == [[2, 3, 4]]
    assert red_indices(view.processed_plot) == [[8, 9, 10, 11]]
    assert not view.artifact_legend.isHidden()
    assert "红色" in view.artifact_legend.text()

    view.set_artifact_settings(ArtifactSettings(enabled=False, threshold_uv=200.0))
    assert view.artifact_legend.isHidden()
    assert red_indices(view.raw_plot) == []
    assert red_indices(view.processed_plot) == []


def test_waveform_renders_an_isolated_threshold_sample(qtbot):
    data = np.zeros((1, 10))
    data[0, 5] = 250e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(["FP1"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_original(raw)
    symbol_items = [
        item
        for item in view.raw_plot.listDataItems()
        if item.opts.get("symbol") == "o"
    ]
    assert len(symbol_items) == 1
    x, _ = symbol_items[0].getData()
    assert x.tolist() == [0.05]


def test_waveform_axis_zoom_controls_are_independent(qtbot):
    raw = mne.io.RawArray(
        np.vstack([
            np.sin(np.linspace(0.0, 40.0, 4000)),
            np.cos(np.linspace(0.0, 40.0, 4000)),
        ]) * 1e-6,
        mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()
    view.set_original(raw)

    x_before, y_before = view.raw_plot.getViewBox().viewRange()
    view.zoom_x_in()
    x_after, y_after_x = view.raw_plot.getViewBox().viewRange()
    assert x_after[1] - x_after[0] < x_before[1] - x_before[0]
    assert y_after_x == pytest.approx(y_before)

    view.zoom_y_in()
    x_after_y, y_after = view.raw_plot.getViewBox().viewRange()
    assert x_after_y == pytest.approx(x_after)
    assert y_after[1] - y_after[0] < y_after_x[1] - y_after_x[0]


def test_batch_cancel_restores_controls(qtbot, tmp_path):
    page = BatchPage()
    qtbot.addWidget(page)
    page.files = [tmp_path / "recording.fif"]
    page._set_busy(True)
    page._task_cancelled("批量任务已取消")
    assert page.start_button.isEnabled()
    assert not page.cancel_button.isEnabled()
    assert "已取消" in page.summary.text()
