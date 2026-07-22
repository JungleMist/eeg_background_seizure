from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from eeg_bg.application.artifacts import artifact_mask
from eeg_bg.application.models import ArtifactSettings

from .theme import COLORS


class ChannelSelectionMenu(QMenu):
    """Keep channel-selection actions open for consecutive changes."""

    def mouseReleaseEvent(self, event) -> None:
        action = self.actionAt(event.position().toPoint())
        if action is not None and action.property("keepMenuOpen"):
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WaveformView(QWidget):
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw = None
        self._processed = None
        self._processed_start = 0.0
        self._visible_channels: list[str] = []
        self._channel_actions: dict[str, QAction] = {}
        self._base_y_ranges: dict[object, tuple[float, float]] = {}
        self._artifact_settings = ArtifactSettings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        tools = QHBoxLayout()
        self.channel_button = QPushButton("通道 0/0")
        self.channel_menu = ChannelSelectionMenu(self)
        self.channel_button.setMenu(self.channel_menu)
        self.reset_button = QPushButton("恢复全局视图")
        self.reset_button.clicked.connect(self.reset_view)
        tools.addWidget(self.channel_button)
        tools.addWidget(self.reset_button)
        self.artifact_legend = QLabel()
        self.artifact_legend.setObjectName("ArtifactLegend")
        self.artifact_legend.setStyleSheet(f"color: {COLORS['danger']};")
        tools.addWidget(self.artifact_legend)
        tools.addStretch(1)
        layout.addLayout(tools)

        axis_tools = QHBoxLayout()
        axis_tools.addStretch(1)
        axis_tools.addWidget(QLabel("时间轴 X"))
        self.x_zoom_out_button = self._zoom_button(
            "−", "缩小时间轴", "显示更长的时间范围", self.zoom_x_out
        )
        self.x_zoom_in_button = self._zoom_button(
            "+", "放大时间轴", "查看更短、更清晰的时间范围", self.zoom_x_in
        )
        axis_tools.addWidget(self.x_zoom_out_button)
        axis_tools.addWidget(self.x_zoom_in_button)
        axis_tools.addSpacing(8)
        axis_tools.addWidget(QLabel("振幅轴 Y"))
        self.y_zoom_out_button = self._zoom_button(
            "−", "缩小振幅轴", "显示更大的纵向范围", self.zoom_y_out
        )
        self.y_zoom_in_button = self._zoom_button(
            "+", "放大振幅轴", "放大通道波形和纵向间距", self.zoom_y_in
        )
        axis_tools.addWidget(self.y_zoom_out_button)
        axis_tools.addWidget(self.y_zoom_in_button)
        layout.addLayout(axis_tools)

        pg.setConfigOptions(antialias=False, background=COLORS["canvas"], foreground=COLORS["muted"])
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setAccessibleName("原始与预处理 EEG 双轨波形")
        self.graphics.setAccessibleDescription("上下波形共享时间轴；选区也可通过右侧起点和终点字段调整")
        self.raw_plot = self.graphics.addPlot(row=0, col=0, title="原始 EEG · 源采样率")
        self.processed_plot = self.graphics.addPlot(row=1, col=0, title="预处理后 EEG")
        self.processed_plot.setXLink(self.raw_plot)
        for plot in (self.raw_plot, self.processed_plot):
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setLabel("bottom", "时间", units="s")
            plot.setMouseEnabled(x=True, y=True)
            plot.getViewBox().setBorder(pg.mkPen(COLORS["border"]))

        self.region = pg.LinearRegionItem(
            values=(0.0, 20.0),
            brush=pg.mkBrush(33, 184, 166, 35),
            pen=pg.mkPen(COLORS["raw"], width=1),
            hoverPen=pg.mkPen(COLORS["text"], width=1),
        )
        self.region.setZValue(10)
        self.raw_plot.addItem(self.region)
        self.region.sigRegionChangeFinished.connect(self._emit_selection)

        self.raw_cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS["muted"], width=1))
        self.processed_cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS["muted"], width=1))
        self.raw_plot.addItem(self.raw_cursor, ignoreBounds=True)
        self.processed_plot.addItem(self.processed_cursor, ignoreBounds=True)
        self._mouse_proxy = pg.SignalProxy(
            self.graphics.scene().sigMouseMoved, rateLimit=30, slot=self._mouse_moved
        )
        layout.addWidget(self.graphics, 1)
        self._update_artifact_legend()

    def _zoom_button(self, text: str, name: str, tooltip: str, callback) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("AxisZoomButton")
        button.setText(text)
        button.setAccessibleName(name)
        button.setToolTip(tooltip)
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(350)
        button.setAutoRepeatInterval(100)
        button.clicked.connect(callback)
        return button

    def _emit_selection(self):
        start, stop = sorted(self.region.getRegion())
        self.selectionChanged.emit(float(start), float(stop))

    def _mouse_moved(self, event):
        pos = event[0]
        if self.raw_plot.sceneBoundingRect().contains(pos):
            x = self.raw_plot.vb.mapSceneToView(pos).x()
        elif self.processed_plot.sceneBoundingRect().contains(pos):
            x = self.processed_plot.vb.mapSceneToView(pos).x()
        else:
            return
        self.raw_cursor.setPos(x)
        self.processed_cursor.setPos(x)

    def set_original(self, raw):
        self._raw = raw
        self._processed = None
        self._visible_channels = list(raw.ch_names)
        self._rebuild_channel_menu()
        duration = raw.n_times / float(raw.info["sfreq"])
        self.region.setBounds((0.0, duration))
        self.region.setRegion((0.0, min(20.0, duration)))
        self.refresh()
        self.reset_view()

    def set_processed(self, raw, start_sec: float):
        self._processed = raw
        self._processed_start = float(start_sec)
        self.refresh()

    def set_artifact_settings(self, settings: ArtifactSettings) -> None:
        settings.validate()
        self._artifact_settings = settings
        self._update_artifact_legend()
        self.refresh()

    def _update_artifact_legend(self) -> None:
        settings = self._artifact_settings
        self.artifact_legend.setText(
            f"● 红色 = |振幅| > {settings.threshold_uv:g} µV"
        )
        self.artifact_legend.setVisible(settings.enabled)

    def clear(self):
        self._raw = None
        self._processed = None
        self._base_y_ranges.clear()
        self.raw_plot.clear()
        self.processed_plot.clear()

    def _rebuild_channel_menu(self):
        self.channel_menu.clear()
        self._channel_actions.clear()
        if self._raw is None:
            return
        select_all = self.channel_menu.addAction("显示全部")
        select_all.setProperty("keepMenuOpen", True)
        select_all.triggered.connect(self._select_all_channels)
        clear_all = self.channel_menu.addAction("取消显示全部")
        clear_all.setProperty("keepMenuOpen", True)
        clear_all.triggered.connect(self._clear_all_channels)
        self.channel_menu.addSeparator()
        for name in self._raw.ch_names:
            action = self.channel_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(True)
            action.setProperty("keepMenuOpen", True)
            action.toggled.connect(self._channel_visibility_changed)
            self._channel_actions[name] = action
        self._update_channel_label()

    def _select_all_channels(self):
        self._set_all_channels_visible(True)

    def _clear_all_channels(self):
        self._set_all_channels_visible(False)

    def _set_all_channels_visible(self, visible: bool) -> None:
        for action in self._channel_actions.values():
            with QSignalBlocker(action):
                action.setChecked(visible)
        self._channel_visibility_changed()

    def _channel_visibility_changed(self):
        self._visible_channels = [
            name for name, action in self._channel_actions.items() if action.isChecked()
        ]
        self._update_channel_label()
        self.refresh()

    def _update_channel_label(self):
        total = len(self._channel_actions)
        self.channel_button.setText(f"通道 {len(self._visible_channels)}/{total}")

    @staticmethod
    def _plot_raw(
        plot,
        raw,
        channels: list[str],
        color: str,
        start_sec: float,
        artifact_settings: ArtifactSettings,
    ):
        plot.clear()
        if raw is None or not channels:
            return None
        available = [channel for channel in channels if channel in raw.ch_names]
        if not available:
            return None
        data_uv = raw.get_data(picks=available) * 1e6
        sfreq = float(raw.info["sfreq"])
        times = start_sec + np.arange(data_uv.shape[1], dtype=float) / sfreq
        scale = float(np.nanpercentile(np.abs(data_uv), 98))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        spacing = 2.4 * scale
        ticks = []
        for index, (name, signal) in enumerate(zip(available, data_uv)):
            offset = (len(available) - 1 - index) * spacing
            item = plot.plot(times, signal + offset, pen=pg.mkPen(color, width=1))
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            if artifact_settings.enabled:
                mask = artifact_mask(signal, artifact_settings.threshold_uv)
                if np.any(mask):
                    overlay = np.where(mask, signal + offset, np.nan)
                    artifact_item = plot.plot(
                        times,
                        overlay,
                        pen=pg.mkPen(COLORS["danger"], width=2),
                        connect="finite",
                    )
                    artifact_item.setDownsampling(auto=True, method="peak")
                    artifact_item.setClipToView(True)
                    previous = np.r_[False, mask[:-1]]
                    following = np.r_[mask[1:], False]
                    isolated = mask & ~previous & ~following
                    if np.any(isolated):
                        plot.plot(
                            times[isolated],
                            signal[isolated] + offset,
                            pen=None,
                            symbol="o",
                            symbolSize=4,
                            symbolPen=pg.mkPen(COLORS["danger"]),
                            symbolBrush=pg.mkBrush(COLORS["danger"]),
                        )
            ticks.append((offset, name))
        plot.getAxis("left").setTicks([ticks])
        y_range = (-spacing, len(available) * spacing)
        plot.setYRange(*y_range, padding=0.02)
        return y_range

    def refresh(self):
        raw_range = self._plot_raw(
            self.raw_plot,
            self._raw,
            self._visible_channels,
            COLORS["raw"],
            0.0,
            self._artifact_settings,
        )
        processed_range = self._plot_raw(
            self.processed_plot,
            self._processed,
            self._visible_channels,
            COLORS["processed"],
            self._processed_start,
            self._artifact_settings,
        )
        self._base_y_ranges = {
            plot: y_range
            for plot, y_range in (
                (self.raw_plot, raw_range),
                (self.processed_plot, processed_range),
            )
            if y_range is not None
        }
        # Plot.clear removes overlays, so restore the shared selection/cursors.
        self.raw_plot.addItem(self.region)
        self.raw_plot.addItem(self.raw_cursor, ignoreBounds=True)
        self.processed_plot.addItem(self.processed_cursor, ignoreBounds=True)

    def reset_view(self):
        if self._raw is None:
            return
        duration = self._raw.n_times / float(self._raw.info["sfreq"])
        self.raw_plot.getViewBox().setLimits(
            xMin=0.0,
            xMax=duration,
            minXRange=max(0.05, 2.0 / float(self._raw.info["sfreq"])),
        )
        self.raw_plot.setXRange(0.0, duration, padding=0.01)
        for plot, y_range in self._base_y_ranges.items():
            plot.setYRange(*y_range, padding=0.02)

    def _zoom_x(self, factor: float) -> None:
        if self._raw is None:
            return
        duration = self._raw.n_times / float(self._raw.info["sfreq"])
        x_min, x_max = self.raw_plot.getViewBox().viewRange()[0]
        span = min(duration, max(0.05, (x_max - x_min) * factor))
        center = (x_min + x_max) / 2.0
        start = max(0.0, min(center - span / 2.0, duration - span))
        self.raw_plot.setXRange(start, start + span, padding=0.0)

    def zoom_x_in(self) -> None:
        self._zoom_x(0.5)

    def zoom_x_out(self) -> None:
        self._zoom_x(2.0)

    def _zoom_y(self, factor: float) -> None:
        for plot in self._base_y_ranges:
            y_min, y_max = plot.getViewBox().viewRange()[1]
            center = (y_min + y_max) / 2.0
            half_span = max(1e-6, (y_max - y_min) * factor / 2.0)
            plot.setYRange(center - half_span, center + half_span, padding=0.0)

    def zoom_y_in(self) -> None:
        self._zoom_y(0.67)

    def zoom_y_out(self) -> None:
        self._zoom_y(1.5)
