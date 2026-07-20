from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import COLORS


class WaveformView(QWidget):
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw = None
        self._processed = None
        self._processed_start = 0.0
        self._visible_channels: list[str] = []
        self._channel_actions: dict[str, QAction] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        tools = QHBoxLayout()
        self.channel_button = QPushButton("通道 0/0")
        self.channel_menu = QMenu(self)
        self.channel_button.setMenu(self.channel_menu)
        self.reset_button = QPushButton("恢复全局视图")
        self.reset_button.clicked.connect(self.reset_view)
        tools.addWidget(self.channel_button)
        tools.addWidget(self.reset_button)
        tools.addStretch(1)
        layout.addLayout(tools)

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

    def clear(self):
        self._raw = None
        self._processed = None
        self.raw_plot.clear()
        self.processed_plot.clear()

    def _rebuild_channel_menu(self):
        self.channel_menu.clear()
        self._channel_actions.clear()
        if self._raw is None:
            return
        select_all = self.channel_menu.addAction("显示全部")
        select_all.triggered.connect(self._select_all_channels)
        self.channel_menu.addSeparator()
        for name in self._raw.ch_names:
            action = self.channel_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(self._channel_visibility_changed)
            self._channel_actions[name] = action
        self._update_channel_label()

    def _select_all_channels(self):
        for action in self._channel_actions.values():
            action.setChecked(True)

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
    def _plot_raw(plot, raw, channels: list[str], color: str, start_sec: float):
        plot.clear()
        if raw is None or not channels:
            return
        available = [channel for channel in channels if channel in raw.ch_names]
        if not available:
            return
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
            ticks.append((offset, name))
        plot.getAxis("left").setTicks([ticks])
        plot.setYRange(-spacing, len(available) * spacing, padding=0.02)

    def refresh(self):
        self._plot_raw(
            self.raw_plot, self._raw, self._visible_channels, COLORS["raw"], 0.0
        )
        self._plot_raw(
            self.processed_plot,
            self._processed,
            self._visible_channels,
            COLORS["processed"],
            self._processed_start,
        )
        # Plot.clear removes overlays, so restore the shared selection/cursors.
        self.raw_plot.addItem(self.region)
        self.raw_plot.addItem(self.raw_cursor, ignoreBounds=True)
        self.processed_plot.addItem(self.processed_cursor, ignoreBounds=True)

    def reset_view(self):
        if self._raw is None:
            return
        duration = self._raw.n_times / float(self._raw.info["sfreq"])
        self.raw_plot.setXRange(0.0, duration, padding=0.01)
