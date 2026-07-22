import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QDialog, QMessageBox, QStyle, QStyleOptionSpinBox

from eeg_bg.application.models import (
    ArtifactSettings,
    ExtractionMode,
    ProcessingMethod,
    ProcessingSpec,
    RecordingEvent,
    RecordingInfo,
    RecordingSidecars,
    WienerMode,
)
from eeg_bg.application.ern_comparison import ErnComparisonResult, ErnWaveform
from eeg_bg.gui.branding import APPLICATION_NAME, PRODUCT_NAME
from eeg_bg.gui.channel_groups import (
    ChannelGroupEditorDialog,
    ChannelGroupPresetStore,
)
from eeg_bg.gui.ern_comparison import ErnComparisonDialog
from eeg_bg.gui.events import RecordingInspectorPanel
from eeg_bg.gui.main_window import MainWindow
from eeg_bg.gui.pages import BatchPage, PreviewPage
from eeg_bg.gui.parameters import ArtifactSettingsStore, ParameterPanel
from eeg_bg.gui.theme import COLORS, stylesheet
from eeg_bg.gui.waveform import WaveformView
from eeg_bg.gui.workers import ErnComparisonWorker


def _ern_result(tmp_path: Path) -> ErnComparisonResult:
    times = np.linspace(-600.0, 400.0, 11)
    waveforms = {
        method: ErnWaveform(
            times_ms=times,
            incorrect_mean_uv=np.linspace(-2.0, 3.0, 11) + index,
            difference_mean_uv=np.linspace(-1.0, 2.0, 11) + index,
            incorrect_sd_uv=np.ones(11) * 2.0,
            difference_sd_uv=np.ones(11) * 3.0,
        )
        for index, method in enumerate(("raw", "ica", "wiener"))
    }
    return ErnComparisonResult(
        source=tmp_path / "sub-001_task-ERN_eeg.set",
        waveforms=waveforms,
        processing_spec=ProcessingSpec(
            wiener_mode=WienerMode.PHASEGATED,
            coherence_threshold=0.15,
            phase_gate_threshold_rad=0.1,
        ),
        n_paired_trials=20,
        n_epochs=18,
        n_correct=14,
        n_incorrect=4,
        rejected_epochs=2,
        ica_excluded_components=(1,),
        wiener_diagnostics={"windows": 3},
        baseline_ms=(-400.0, -200.0),
        peak_window_ms=(0.0, 150.0),
    )


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


def test_ern_comparison_dialog_draws_three_methods_and_trial_context(qtbot, tmp_path):
    dialog = ErnComparisonDialog(_ern_result(tmp_path))
    qtbot.addWidget(dialog)
    dialog.show()

    assert set(dialog.method_curves["incorrect"]) == {"raw", "ica", "wiener"}
    assert set(dialog.method_curves["difference"]) == {"raw", "ica", "wiener"}
    assert dialog.incorrect_plot.getViewBox().state["yInverted"]
    assert "sub-001_task-ERN_eeg.set" in dialog.windowTitle()
    assert "FCz" in dialog.graphics.accessibleName()


def test_preview_enables_and_starts_ern_comparison_with_current_parameters(
    qtbot, tmp_path, monkeypatch
):
    page = PreviewPage(
        settings=QSettings(str(tmp_path / "ern-preview.ini"), QSettings.IniFormat)
    )
    qtbot.addWidget(page)
    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["FCz"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    source = tmp_path / "sub-001_task-ERN_eeg.set"
    info = RecordingInfo(
        path=source,
        format="set",
        ch_names=["FCz"],
        sfreq=100.0,
        duration_sec=10.0,
        n_times=1000,
        sidecars=RecordingSidecars(
            eeg={"TaskName": "ERN"},
            events=[
                RecordingEvent(1.0, trial_type="stimulus", value="11"),
                RecordingEvent(1.4, trial_type="response", value="111"),
            ],
        ),
    )
    page._loaded((info, raw, []))
    page._thread_finished()
    assert page.ern_compare_button.isEnabled()

    page.parameters.wiener_mode.setCurrentIndex(
        page.parameters.wiener_mode.findData(WienerMode.PHASEGATED)
    )
    page.parameters.coherence.setValue(0.45)
    page.parameters.gate.setValue(0.1)
    captured = []

    def capture(worker, on_finished, on_failed=None):
        captured.append(worker)
        return True

    monkeypatch.setattr(page, "_start_worker", capture)
    page.compare_ern()

    assert isinstance(captured[0], ErnComparisonWorker)
    assert captured[0].processing.wiener_mode is WienerMode.PHASEGATED
    assert captured[0].processing.coherence_threshold == 0.45
    assert captured[0].processing.phase_gate_threshold_rad == 0.1


def test_recording_inspector_filters_and_navigates_events(qtbot, tmp_path):
    panel = RecordingInspectorPanel()
    qtbot.addWidget(panel)
    events = [
        RecordingEvent(1.0, trial_type="stimulus", value="11"),
        RecordingEvent(2.0, trial_type="stimulus", value="12"),
        RecordingEvent(2.4, trial_type="response", value="111", sample=2458),
        RecordingEvent(3.0, trial_type="stimulus", value="11"),
    ]
    info = RecordingInfo(
        path=tmp_path / "subject.set",
        format="set",
        ch_names=["FCz"],
        sfreq=1024.0,
        duration_sec=10.0,
        n_times=10240,
        sidecars=RecordingSidecars(
            eeg={"TaskName": "ERN", "Manufacturer": "Biosemi"},
            channels=[{"name": "FCz", "type": "EEG"}],
            events=events,
        ),
    )
    panel.set_recording(info)
    panel.event_type.setCurrentIndex(panel.event_type.findData("stimulus"))
    panel.event_value.setCurrentIndex(panel.event_value.findData("11"))

    with qtbot.waitSignal(panel.eventActivated, timeout=1000) as first:
        panel.next_event()
    with qtbot.waitSignal(panel.eventActivated, timeout=1000) as second:
        panel.next_event()

    assert first.args[0].onset_sec == 1.0
    assert second.args[0].onset_sec == 3.0
    assert "2/2" in panel.event_status.text()
    assert "ERN" in panel.metadata_summary.text()


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
    assert panel.extraction_form.isRowVisible(panel.current_window_button)
    assert not panel.current_window_button.isEnabled()
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
    assert not panel.extraction_form.isRowVisible(panel.current_window_button)
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
    panel.set_current_window_available(True)
    assert panel.extraction_form.isRowVisible(panel.current_window_button)
    assert panel.current_window_button.isEnabled()
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.WIENER))
    assert panel.base_form.isRowVisible(panel.analysis_window)
    assert panel.method_form.isRowVisible(panel.wiener_mode)
    assert panel.method_form.isRowVisible(panel.coherence)
    assert panel.method_form.isRowVisible(panel.channel_groups_button)
    assert not panel.channel_groups_button.isEnabled()
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


def test_channel_group_editor_adds_and_removes_groups_and_current_eeg_channels(
    qtbot,
):
    dialog = ChannelGroupEditorDialog(
        ["FP1", "FP2", "FCz"],
        (("FP1", "FP2"),),
    )
    qtbot.addWidget(dialog)

    dialog.add_group()
    assert len(dialog.groups()) == 2
    dialog.channel_combo.setCurrentIndex(dialog.channel_combo.findText("FCz"))
    dialog.add_channel()
    dialog.channel_combo.setCurrentIndex(dialog.channel_combo.findText("FP1"))
    dialog.add_channel()
    assert dialog.groups()[1] == ("FCz", "FP1")

    dialog.group_list.setCurrentRow(0)
    dialog.remove_group()
    assert dialog.groups() == (("FCz", "FP1"),)
    dialog.channel_list.setCurrentRow(1)
    dialog.remove_channel()
    dialog.channel_combo.setCurrentIndex(dialog.channel_combo.findText("FP2"))
    dialog.add_channel()

    assert dialog.groups() == (("FCz", "FP2"),)
    assert {
        dialog.channel_combo.itemText(index)
        for index in range(dialog.channel_combo.count())
    } <= {"FP1", "FP2", "FCz"}
    dialog._accept_valid()
    assert dialog.result() == QDialog.Accepted


def test_channel_group_presets_persist_save_load_filter_and_delete(
    qtbot, tmp_path, monkeypatch
):
    settings_path = tmp_path / "presets.ini"
    store = ChannelGroupPresetStore(
        QSettings(str(settings_path), QSettings.IniFormat)
    )
    store.save(
        "ERP 前中线",
        (("FP1", "FP2"), ("FCz", "Cz")),
    )
    restored = ChannelGroupPresetStore(
        QSettings(str(settings_path), QSettings.IniFormat)
    )
    assert restored.get("ERP 前中线") == (
        ("FP1", "FP2"),
        ("FCz", "Cz"),
    )

    dialog = ChannelGroupEditorDialog(
        ["FP1", "FP2", "FCz"],
        (("FP1", "FP2"),),
        preset_store=store,
    )
    qtbot.addWidget(dialog)
    notices = []
    monkeypatch.setattr(
        "eeg_bg.gui.channel_groups.QMessageBox.information",
        lambda *args: notices.append(args[-1]),
    )
    dialog.preset_combo.setCurrentText("ERP 前中线")
    dialog.load_preset()
    assert dialog.groups() == (("FP1", "FP2"), ("FCz",))
    assert "Cz" in notices[0]

    valid_dialog = ChannelGroupEditorDialog(
        ["FP1", "FP2", "FCz"],
        (("FP1", "FCz"),),
        preset_store=store,
    )
    qtbot.addWidget(valid_dialog)
    monkeypatch.setattr(
        "eeg_bg.gui.channel_groups.QInputDialog.getText",
        lambda *args, **kwargs: ("自定义前额", True),
    )
    valid_dialog.save_preset()
    assert store.get("自定义前额") == (("FP1", "FCz"),)

    monkeypatch.setattr(
        "eeg_bg.gui.channel_groups.QMessageBox.question",
        lambda *args: QMessageBox.Yes,
    )
    valid_dialog.preset_combo.setCurrentText("自定义前额")
    valid_dialog.delete_preset()
    assert store.get("自定义前额") is None


def test_preview_ecmad_groups_are_filtered_to_current_eeg_and_enter_spec(qtbot):
    panel = ParameterPanel(allow_selection=True)
    qtbot.addWidget(panel)
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.WIENER))
    panel.set_channel_context(
        ["FP1", "FP2", "FCz"],
        (("FP1", "FP2"), ("F7", "T3")),
    )

    assert panel.channel_groups_button.isEnabled()
    assert panel.channel_groups_button.text() == "编辑（1 组）"
    assert panel.processing_spec().channel_groups == (("FP1", "FP2"),)


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


def test_preview_uses_visible_window_as_interactive_selection(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "preview.ini"), QSettings.IniFormat)
    page = PreviewPage(settings=settings)
    qtbot.addWidget(page)
    assert not page.parameters.current_window_button.isEnabled()

    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["FCz"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    info = RecordingInfo(
        path=tmp_path / "sample.fif",
        format="fif",
        sfreq=100.0,
        ch_names=("FCz",),
        n_times=1000,
        duration_sec=10.0,
    )
    page._loaded((info, raw, []))
    page.waveform.raw_plot.setXRange(2.25, 6.75, padding=0.0)

    with qtbot.waitSignal(page.statusChanged, timeout=1000) as status:
        qtbot.mouseClick(page.parameters.current_window_button, Qt.LeftButton)

    assert page.parameters.start_sec.value() == pytest.approx(2.25)
    assert page.parameters.stop_sec.value() == pytest.approx(6.75)
    assert page.waveform.region.getRegion() == pytest.approx((2.25, 6.75))
    assert "2.25–6.75 s" in status.args[0]


def test_preview_replaces_result_and_unlocks_only_after_thread_cleanup(
    qtbot, tmp_path
):
    page = PreviewPage(
        settings=QSettings(str(tmp_path / "refresh.ini"), QSettings.IniFormat)
    )
    qtbot.addWidget(page)
    info = mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg")
    times = np.arange(1000, dtype=float) / 100.0
    original = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 20e-6,
        info,
        verbose=False,
    )
    first = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 10e-6,
        info,
        verbose=False,
    )
    second = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 2e-6,
        info,
        verbose=False,
    )
    page.source_path = tmp_path / "source-raw.fif"
    page.waveform.set_original(original)

    page._set_busy(True)
    assert not page.parameters.isEnabled()
    page._processed(
        SimpleNamespace(
            processed_segments=[SimpleNamespace(raw=first, start_sec=0.0)],
            processing_spec=ProcessingSpec(method=ProcessingMethod.BASIC),
            warnings=[],
        )
    )
    first_curve = page.waveform.processed_plot.listDataItems()[0].getData()[1].copy()
    assert not page.process_button.isEnabled()
    page._thread_finished()
    assert page.process_button.isEnabled()
    assert page.parameters.isEnabled()

    page._set_busy(True)
    page._processed(
        SimpleNamespace(
            processed_segments=[SimpleNamespace(raw=second, start_sec=0.0)],
            processing_spec=ProcessingSpec(
                method=ProcessingMethod.WIENER,
                wiener_mode=WienerMode.PHASEGATED,
                coherence_threshold=0.45,
                phase_gate_threshold_rad=0.1,
            ),
            warnings=[],
        )
    )
    second_curve = page.waveform.processed_plot.listDataItems()[0].getData()[1]

    assert not np.array_equal(first_curve, second_curve)
    assert "ECMAD phasegated" in page.waveform.processed_plot.titleLabel.text
    assert "coherence 0.45" in page.waveform.processed_plot.titleLabel.text
    assert not page.process_button.isEnabled()
    page._thread_finished()
    assert page.process_button.isEnabled()


def test_preview_runs_two_parameter_sets_and_renders_the_latest_result(
    qtbot, tmp_path
):
    sfreq = 125.0
    times = np.arange(1000, dtype=float) / sfreq
    data = np.vstack(
        [
            20e-6 * np.sin(2 * np.pi * 4 * times)
            + 12e-6 * np.sin(2 * np.pi * 25 * times),
            16e-6 * np.cos(2 * np.pi * 4 * times)
            + 10e-6 * np.cos(2 * np.pi * 25 * times),
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["FP1", "FP2"], sfreq, ch_types="eeg"),
        verbose=False,
    )
    source = tmp_path / "refresh-raw.fif"
    raw.save(source, overwrite=True, verbose=False)

    page = PreviewPage(
        settings=QSettings(str(tmp_path / "threaded.ini"), QSettings.IniFormat)
    )
    qtbot.addWidget(page)
    page.source_path = source
    page.waveform.set_original(raw)
    page.parameters.extraction_mode.setCurrentIndex(
        page.parameters.extraction_mode.findData(ExtractionMode.CONTINUOUS)
    )
    page.parameters.method.setCurrentIndex(
        page.parameters.method.findData(ProcessingMethod.BASIC)
    )

    page.parameters.high_hz.setValue(40.0)
    page.process()
    qtbot.waitUntil(lambda: page._thread is None, timeout=15000)
    first_curve = page.waveform.processed_plot.listDataItems()[0].getData()[1].copy()

    page.parameters.high_hz.setValue(10.0)
    page.process()
    qtbot.waitUntil(lambda: page._thread is None, timeout=15000)
    second_curve = page.waveform.processed_plot.listDataItems()[0].getData()[1]

    assert page.current_result.processing_spec.bandpass_high_hz == 10.0
    assert not np.allclose(first_curve, second_curve, atol=1e-3)
    assert "0.5–10 Hz" in page.waveform.processed_plot.titleLabel.text


def test_waveform_focuses_and_marks_selected_event(qtbot):
    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["FCz"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_original(raw)
    view.focus_event(RecordingEvent(6.0, trial_type="stimulus", value="11"))

    x_range = view.raw_plot.getViewBox().viewRange()[0]
    assert view.raw_event_line.isVisible()
    assert view.processed_event_line.isVisible()
    assert view.raw_event_line.value() == 6.0
    assert x_range == pytest.approx([4.0, 8.0])

    view.refresh()
    assert view.raw_event_line.isVisible()
    assert view.raw_event_line.value() == 6.0

    view.set_original(raw.copy())
    assert not view.raw_event_line.isVisible()
    assert not view.processed_event_line.isVisible()


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


def test_waveform_uses_one_fixed_gain_and_channel_spacing(qtbot):
    times = np.linspace(0.0, 4.0 * np.pi, 1000)
    raw_data = np.vstack([
        5e-3 + 20e-6 * np.sin(times),
        -3e-3 + 10e-6 * np.cos(times),
    ])
    processed_data = np.vstack([
        2e-3 + 5e-6 * np.sin(times),
        -1e-3 + 2e-6 * np.cos(times),
    ])
    info = mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    processed = mne.io.RawArray(processed_data, info, verbose=False)
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_artifact_settings(ArtifactSettings(enabled=False))
    view.set_original(raw)
    fixed_scale = view._display_scale_uv
    fixed_spacing = view._channel_spacing_uv

    view.set_processed(processed, 0.0)

    assert view._display_scale_uv == fixed_scale
    assert view._channel_spacing_uv == fixed_spacing
    assert view._base_y_ranges[view.raw_plot] == view._base_y_ranges[view.processed_plot]
    assert view.raw_plot.getViewBox().viewRange()[1] == pytest.approx(
        view.processed_plot.getViewBox().viewRange()[1]
    )
    raw_curves = view.raw_plot.listDataItems()
    processed_curves = view.processed_plot.listDataItems()
    assert np.ptp(raw_curves[0].getData()[1]) == pytest.approx(40.0, abs=0.1)
    assert np.ptp(processed_curves[0].getData()[1]) == pytest.approx(10.0, abs=0.1)
    assert np.median(raw_curves[0].getData()[1]) == pytest.approx(
        fixed_spacing, abs=0.1
    )

    view.raw_plot.setYRange(-50.0, 50.0, padding=0.0)
    assert view.processed_plot.getViewBox().viewRange()[1] == pytest.approx(
        [-50.0, 50.0]
    )
    view._channel_actions["FP2"].setChecked(False)
    assert view._channel_spacing_uv == fixed_spacing
    assert view._base_y_ranges[view.raw_plot] == view._base_y_ranges[view.processed_plot]


def test_waveform_preserves_amplitude_view_after_each_processing_result(qtbot):
    times = np.linspace(0.0, 4.0 * np.pi, 1000)
    info = mne.create_info(["FP1", "FP2"], 100.0, ch_types="eeg")
    original = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 20e-6,
        info,
        verbose=False,
    )
    first = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 8e-6,
        info,
        verbose=False,
    )
    second = mne.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]) * 2e-6,
        info,
        verbose=False,
    )
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_original(original)
    view.raw_plot.setYRange(-35.0, 75.0, padding=0.0)
    expected_range = tuple(view.raw_plot.getViewBox().viewRange()[1])

    view.set_processed(first, 0.0, "ICA")
    assert view.raw_plot.getViewBox().viewRange()[1] == pytest.approx(expected_range)
    assert view.processed_plot.getViewBox().viewRange()[1] == pytest.approx(
        expected_range
    )

    view.set_processed(second, 0.0, "ECMAD phasegated")
    assert view.raw_plot.getViewBox().viewRange()[1] == pytest.approx(expected_range)
    assert view.processed_plot.getViewBox().viewRange()[1] == pytest.approx(
        expected_range
    )


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
