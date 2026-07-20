import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt

from eeg_bg.application.models import ProcessingMethod, WienerMode
from eeg_bg.gui.main_window import MainWindow
from eeg_bg.gui.pages import BatchPage
from eeg_bg.gui.parameters import ParameterPanel
from eeg_bg.gui.waveform import WaveformView


def test_main_window_has_two_workspaces(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.stack.count() == 2
    qtbot.mouseClick(window.batch_nav, Qt.LeftButton)
    assert window.stack.currentIndex() == 1


def test_parameter_panel_enables_gate_only_for_gated_modes(qtbot):
    panel = ParameterPanel(allow_selection=True)
    qtbot.addWidget(panel)
    panel.method.setCurrentIndex(panel.method.findData(ProcessingMethod.WIENER))
    panel.wiener_mode.setCurrentIndex(panel.wiener_mode.findData(WienerMode.FREQUENCY))
    assert not panel.gate.isEnabled()
    panel.wiener_mode.setCurrentIndex(panel.wiener_mode.findData(WienerMode.PHASEGATED))
    assert panel.gate.isEnabled()


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


def test_batch_cancel_restores_controls(qtbot, tmp_path):
    page = BatchPage()
    qtbot.addWidget(page)
    page.files = [tmp_path / "recording.fif"]
    page._set_busy(True)
    page._task_cancelled("批量任务已取消")
    assert page.start_button.isEnabled()
    assert not page.cancel_button.isEnabled()
    assert "已取消" in page.summary.text()
