from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from eeg_bg.application.ern_comparison import ErnComparisonResult, METHODS

from .theme import COLORS


METHOD_LABELS = {"raw": "Raw", "ica": "标准 ICA", "wiener": "ECMAD"}
METHOD_COLORS = {
    "raw": COLORS["erp_raw"],
    "ica": COLORS["erp_ica"],
    "wiener": COLORS["erp_wiener"],
}


def _alpha_brush(color: str, alpha: int):
    value = pg.mkColor(color)
    value.setAlpha(alpha)
    return pg.mkBrush(value)


class ErnComparisonDialog(QDialog):
    def __init__(self, result: ErnComparisonResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setObjectName("AppRoot")
        self.method_curves: dict[str, dict[str, object]] = {
            "incorrect": {},
            "difference": {},
        }
        self.setWindowTitle(f"ERN 三方法叠加 · {result.source.name}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setModal(False)
        self.resize(1240, 760)
        self.setMinimumSize(860, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        title = QLabel("FCz · 响应锁定 ERN 三方法叠加")
        title.setObjectName("PageTitle")
        layout.addWidget(title)
        spec = result.processing_spec
        summary = QLabel(
            f"{result.source.name}   ·   保留 {result.n_epochs}/{result.n_paired_trials} 试次"
            f"（正确 {result.n_correct} / 错误 {result.n_incorrect}）   ·   "
            f"{spec.bandpass_low_hz:g}–{spec.bandpass_high_hz:g} Hz @ "
            f"{spec.target_sfreq:g} Hz   ·   ECMAD {spec.wiener_mode.value}, "
            f"coherence={spec.coherence_threshold:.3f}, "
            f"phase={spec.effective_phase_gate_rad:.3f} rad"
        )
        summary.setObjectName("Metadata")
        summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        pg.setConfigOptions(
            antialias=True,
            background=COLORS["canvas"],
            foreground=COLORS["muted"],
        )
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setAccessibleName("FCz Raw、ICA 与 ECMAD 的响应锁定 ERN 叠加图")
        self.incorrect_plot = self.graphics.addPlot(
            row=0, col=0, title="FCz 错误响应试次"
        )
        self.difference_plot = self.graphics.addPlot(
            row=0, col=1, title="FCz ERN 差异波"
        )
        self._configure_plot(
            self.incorrect_plot,
            "振幅 (µV)",
            result.peak_window_ms,
        )
        self._configure_plot(
            self.difference_plot,
            "错误 − 正确 (µV)",
            result.peak_window_ms,
        )
        self.incorrect_plot.addLegend(offset=(12, 12))
        self._draw_waveforms()
        layout.addWidget(self.graphics, 1)

        footer = QHBoxLayout()
        note = QLabel(
            f"阴影：均值 ±1 个试次 SD   ·   灰区："
            f"{result.peak_window_ms[0]:g}–{result.peak_window_ms[1]:g} ms ERN 测量窗   ·   "
            f"基线：{result.baseline_ms[0]:g}–{result.baseline_ms[1]:g} ms   ·   "
            f"标准 ICA（FP1/FP2 自动 EOG）移除组件 "
            f"{list(result.ica_excluded_components) or '无'}"
        )
        note.setObjectName("Muted")
        note.setWordWrap(True)
        footer.addWidget(note, 1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        layout.addLayout(footer)

    @staticmethod
    def _configure_plot(plot, y_label: str, peak_window_ms: tuple[float, float]) -> None:
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel("bottom", "响应锁定时间", units="ms")
        plot.setLabel("left", y_label)
        plot.getViewBox().invertY(True)
        plot.getViewBox().setBorder(pg.mkPen(COLORS["border"]))
        region = pg.LinearRegionItem(
            values=peak_window_ms,
            movable=False,
            brush=_alpha_brush(COLORS["erp_window"], 28),
            pen=pg.mkPen(None),
        )
        region.setZValue(-20)
        plot.addItem(region)
        plot.addItem(
            pg.InfiniteLine(angle=90, pos=0.0, pen=pg.mkPen(COLORS["text"], width=1))
        )
        plot.addItem(
            pg.InfiniteLine(angle=0, pos=0.0, pen=pg.mkPen(COLORS["muted"], width=1))
        )

    def _draw_band(self, plot, x, mean, sd, color: str) -> None:
        lower = pg.PlotCurveItem(x, mean - sd, pen=pg.mkPen(None))
        upper = pg.PlotCurveItem(x, mean + sd, pen=pg.mkPen(None))
        plot.addItem(lower)
        plot.addItem(upper)
        band = pg.FillBetweenItem(
            lower,
            upper,
            brush=_alpha_brush(color, 30),
        )
        band.setZValue(1)
        plot.addItem(band)

    def _draw_waveforms(self) -> None:
        for method in METHODS:
            waveform = self.result.waveforms[method]
            color = METHOD_COLORS[method]
            self._draw_band(
                self.incorrect_plot,
                waveform.times_ms,
                waveform.incorrect_mean_uv,
                waveform.incorrect_sd_uv,
                color,
            )
            self._draw_band(
                self.difference_plot,
                waveform.times_ms,
                waveform.difference_mean_uv,
                waveform.difference_sd_uv,
                color,
            )
            incorrect_curve = self.incorrect_plot.plot(
                waveform.times_ms,
                waveform.incorrect_mean_uv,
                pen=pg.mkPen(color, width=2),
                name=METHOD_LABELS[method],
            )
            difference_curve = self.difference_plot.plot(
                waveform.times_ms,
                waveform.difference_mean_uv,
                pen=pg.mkPen(color, width=2),
            )
            incorrect_curve.setZValue(5)
            difference_curve.setZValue(5)
            self.method_curves["incorrect"][method] = incorrect_curve
            self.method_curves["difference"][method] = difference_curve
        times = self.result.waveforms["raw"].times_ms
        for plot in (self.incorrect_plot, self.difference_plot):
            plot.setXRange(float(times[0]), float(times[-1]), padding=0.01)
