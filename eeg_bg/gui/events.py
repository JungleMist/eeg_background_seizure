from __future__ import annotations

import json

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eeg_bg.application.models import RecordingEvent, RecordingInfo


def _value_sort_key(value: str) -> tuple[bool, int | str]:
    try:
        return False, int(value)
    except ValueError:
        return True, value


class RecordingInspectorPanel(QGroupBox):
    eventActivated = Signal(object)

    def __init__(self, parent=None):
        super().__init__("记录元数据与事件", parent)
        self._info: RecordingInfo | None = None
        self._filtered_events: list[RecordingEvent] = []
        self._current_index = -1

        layout = QVBoxLayout(self)
        self.metadata_summary = QLabel("打开 EEGLAB SET 后读取 BIDS 侧车")
        self.metadata_summary.setObjectName("Muted")
        self.metadata_summary.setWordWrap(True)
        layout.addWidget(self.metadata_summary)

        self.details_button = QPushButton("查看侧车元数据")
        self.details_button.setEnabled(False)
        self.details_button.clicked.connect(self._show_metadata)
        layout.addWidget(self.details_button)

        form = QFormLayout()
        self.event_type = QComboBox()
        self.event_type.setAccessibleName("事件类型")
        self.event_value = QComboBox()
        self.event_value.setAccessibleName("事件码")
        form.addRow("事件类型", self.event_type)
        form.addRow("事件码", self.event_value)
        layout.addLayout(form)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("上一个")
        self.next_button = QPushButton("下一个")
        self.previous_button.clicked.connect(self.previous_event)
        self.next_button.clicked.connect(self.next_event)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)

        self.event_status = QLabel("当前文件没有可筛选事件")
        self.event_status.setObjectName("Muted")
        self.event_status.setWordWrap(True)
        layout.addWidget(self.event_status)

        self.event_type.currentIndexChanged.connect(self._type_changed)
        self.event_value.currentIndexChanged.connect(self._apply_filter)
        self._set_navigation_enabled(False)

    def set_recording(self, info: RecordingInfo) -> None:
        self._info = info
        sidecars = info.sidecars
        eeg = sidecars.eeg
        task = str(eeg.get("TaskName", "未标注任务"))
        manufacturer = str(eeg.get("Manufacturer", "未知设备"))
        model = str(eeg.get("ManufacturersModelName", ""))
        device = " ".join(part for part in (manufacturer, model) if part)
        self.metadata_summary.setText(
            f"{task} · {device} · {len(sidecars.channels)} 个通道条目 · "
            f"{len(sidecars.events)} 个事件"
        )
        self.details_button.setEnabled(bool(sidecars.paths))

        event_types = sorted(
            {event.trial_type for event in sidecars.events if event.trial_type}
        )
        with QSignalBlocker(self.event_type):
            self.event_type.clear()
            self.event_type.addItem(f"全部事件（{len(sidecars.events)}）", None)
            for event_type in event_types:
                count = sum(event.trial_type == event_type for event in sidecars.events)
                self.event_type.addItem(f"{event_type}（{count}）", event_type)
        self._rebuild_values()

    def _type_changed(self) -> None:
        self._rebuild_values()

    def _events_for_type(self) -> list[RecordingEvent]:
        if self._info is None:
            return []
        event_type = self.event_type.currentData()
        return [
            event
            for event in self._info.sidecars.events
            if event_type is None or event.trial_type == event_type
        ]

    def _rebuild_values(self) -> None:
        events = self._events_for_type()
        values = sorted(
            {event.value for event in events if event.value}, key=_value_sort_key
        )
        with QSignalBlocker(self.event_value):
            self.event_value.clear()
            self.event_value.addItem(f"全部事件码（{len(events)}）", None)
            for value in values:
                count = sum(event.value == value for event in events)
                self.event_value.addItem(f"{value}（{count}）", value)
        self._apply_filter()

    def _apply_filter(self) -> None:
        value = self.event_value.currentData()
        self._filtered_events = [
            event
            for event in self._events_for_type()
            if value is None or event.value == value
        ]
        self._current_index = -1
        enabled = bool(self._filtered_events)
        self._set_navigation_enabled(enabled)
        self.event_status.setText(
            f"{len(self._filtered_events)} 个匹配事件 · 使用按钮定位"
            if enabled
            else "当前筛选没有事件"
        )

    def _set_navigation_enabled(self, enabled: bool) -> None:
        self.previous_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

    def previous_event(self) -> None:
        if not self._filtered_events:
            return
        self._current_index = (
            len(self._filtered_events) - 1
            if self._current_index < 0
            else (self._current_index - 1) % len(self._filtered_events)
        )
        self._activate_current()

    def next_event(self) -> None:
        if not self._filtered_events:
            return
        self._current_index = (self._current_index + 1) % len(self._filtered_events)
        self._activate_current()

    def _activate_current(self) -> None:
        event = self._filtered_events[self._current_index]
        sample = f" · sample {event.sample}" if event.sample is not None else ""
        duration = f" · duration {event.duration_sec:g} s" if event.duration_sec else ""
        self.event_status.setText(
            f"{self._current_index + 1}/{len(self._filtered_events)} · "
            f"{event.trial_type or 'event'} / {event.value or 'n/a'} · "
            f"{event.onset_sec:.4f} s{sample}{duration}"
        )
        self.eventActivated.emit(event)

    def _show_metadata(self) -> None:
        if self._info is None:
            return
        sidecars = self._info.sidecars
        payload = {
            "files": {key: str(path) for key, path in sidecars.paths.items()},
            "eeg": sidecars.eeg,
            "coordsystem": sidecars.coordsystem,
            "channels": sidecars.channels,
            "electrodes": sidecars.electrodes,
            "events": [event.fields for event in sidecars.events],
        }
        dialog = QDialog(self)
        dialog.setWindowTitle("EEG BIDS 侧车元数据")
        dialog.resize(760, 600)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        layout.addWidget(text)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()
