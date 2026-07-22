from __future__ import annotations

import json

from PySide6.QtCore import QObject, QSettings, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from eeg_bg.application.models import ProcessingMethod, ProcessingSpec


class ChannelGroupPresetStore(QObject):
    changed = Signal()

    KEY = "global/channel_group_presets"

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings

    def presets(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        encoded = self._settings.value(self.KEY, "{}")
        try:
            payload = json.loads(encoded) if isinstance(encoded, str) else encoded
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        presets: dict[str, tuple[tuple[str, ...], ...]] = {}
        for name, groups in payload.items():
            if not isinstance(name, str) or not isinstance(groups, list):
                continue
            normalized = []
            for group in groups:
                if not isinstance(group, list) or not all(
                    isinstance(channel, str) for channel in group
                ):
                    normalized = []
                    break
                normalized.append(tuple(group))
            if normalized:
                presets[name] = tuple(normalized)
        return presets

    def get(self, name: str) -> tuple[tuple[str, ...], ...] | None:
        return self.presets().get(name)

    def save(self, name: str, groups: tuple[tuple[str, ...], ...]) -> None:
        name = name.strip()
        if not name:
            raise ValueError("预设名称不能为空")
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            channel_groups=groups,
        ).validate()
        presets = self.presets()
        presets[name] = tuple(tuple(group) for group in groups)
        payload = {
            preset_name: [list(group) for group in preset_groups]
            for preset_name, preset_groups in presets.items()
        }
        self._settings.setValue(
            self.KEY,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        self.changed.emit()

    def delete(self, name: str) -> None:
        presets = self.presets()
        if name not in presets:
            return
        del presets[name]
        payload = {
            preset_name: [list(group) for group in preset_groups]
            for preset_name, preset_groups in presets.items()
        }
        self._settings.setValue(
            self.KEY,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        self.changed.emit()


class ChannelGroupEditorDialog(QDialog):
    """Edit ECMAD channel groups using only channels from the current EEG."""

    def __init__(
        self,
        available_channels: list[str] | tuple[str, ...],
        groups: tuple[tuple[str, ...], ...] | list[list[str]],
        preset_store: ChannelGroupPresetStore | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("AppRoot")
        self.setWindowTitle("ECMAD 导联组编辑")
        self.setMinimumSize(680, 430)
        self._available_channels = list(dict.fromkeys(available_channels))
        self._preset_store = preset_store
        available = set(self._available_channels)
        self._groups = [
            [channel for channel in group if channel in available]
            for group in groups
        ]

        root = QVBoxLayout(self)
        title = QLabel("导联组")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        description = QLabel(
            "每组至少两个导联；同一导联可参与多个传导路径。"
            f" 当前 EEG 可用 {len(self._available_channels)} 个导联。"
        )
        description.setObjectName("Muted")
        description.setWordWrap(True)
        root.addWidget(description)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.setAccessibleName("ECMAD 导联组预设")
        preset_row.addWidget(self.preset_combo, 1)
        self.load_preset_button = QPushButton("读取预设")
        self.save_preset_button = QPushButton("保存为预设")
        self.delete_preset_button = QPushButton("删除预设")
        self.delete_preset_button.setObjectName("Danger")
        self.load_preset_button.clicked.connect(self.load_preset)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.delete_preset_button.clicked.connect(self.delete_preset)
        preset_row.addWidget(self.load_preset_button)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        root.addLayout(preset_row)

        columns = QHBoxLayout()
        group_column = QVBoxLayout()
        group_column.addWidget(QLabel("导联组"))
        self.group_list = QListWidget()
        self.group_list.setAccessibleName("ECMAD 导联组列表")
        self.group_list.currentRowChanged.connect(self._group_changed)
        group_column.addWidget(self.group_list, 1)
        group_actions = QHBoxLayout()
        self.add_group_button = QPushButton("新增组")
        self.remove_group_button = QPushButton("删除组")
        self.remove_group_button.setObjectName("Danger")
        self.add_group_button.clicked.connect(self.add_group)
        self.remove_group_button.clicked.connect(self.remove_group)
        group_actions.addWidget(self.add_group_button)
        group_actions.addWidget(self.remove_group_button)
        group_column.addLayout(group_actions)
        columns.addLayout(group_column, 1)

        channel_column = QVBoxLayout()
        channel_column.addWidget(QLabel("当前组导联"))
        self.channel_list = QListWidget()
        self.channel_list.setAccessibleName("当前导联组内的导联")
        self.channel_list.currentRowChanged.connect(self._sync_actions)
        channel_column.addWidget(self.channel_list, 1)
        channel_picker = QHBoxLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.setAccessibleName("当前 EEG 可添加导联")
        self.add_channel_button = QPushButton("添加导联")
        self.add_channel_button.clicked.connect(self.add_channel)
        channel_picker.addWidget(self.channel_combo, 1)
        channel_picker.addWidget(self.add_channel_button)
        channel_column.addLayout(channel_picker)
        self.remove_channel_button = QPushButton("删除选中导联")
        self.remove_channel_button.setObjectName("Danger")
        self.remove_channel_button.clicked.connect(self.remove_channel)
        channel_column.addWidget(self.remove_channel_button)
        columns.addLayout(channel_column, 1)
        root.addLayout(columns, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.buttons.button(QDialogButtonBox.Save).setText("保存导联组")
        self.buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self.buttons.accepted.connect(self._accept_valid)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.preset_combo.currentIndexChanged.connect(self._sync_preset_actions)
        if self._preset_store is not None:
            self._preset_store.changed.connect(self._refresh_presets)
        self._refresh_presets()
        self._refresh_groups(0 if self._groups else -1)

    def groups(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(group) for group in self._groups)

    def add_group(self) -> None:
        self._groups.append([])
        self._refresh_groups(len(self._groups) - 1)

    def remove_group(self) -> None:
        index = self.group_list.currentRow()
        if index < 0:
            return
        del self._groups[index]
        self._refresh_groups(min(index, len(self._groups) - 1))

    def add_channel(self) -> None:
        index = self.group_list.currentRow()
        channel = self.channel_combo.currentText()
        if index < 0 or not channel or channel in self._groups[index]:
            return
        self._groups[index].append(channel)
        self._refresh_groups(index)

    def remove_channel(self) -> None:
        group_index = self.group_list.currentRow()
        channel_index = self.channel_list.currentRow()
        if group_index < 0 or channel_index < 0:
            return
        del self._groups[group_index][channel_index]
        self._refresh_groups(group_index)

    def save_preset(self) -> None:
        if self._preset_store is None:
            return
        try:
            ProcessingSpec(
                method=ProcessingMethod.WIENER,
                channel_groups=self.groups(),
            ).validate()
        except ValueError as exc:
            QMessageBox.warning(self, "导联组需要调整", str(exc))
            return
        suggested = self.preset_combo.currentText()
        name, accepted = QInputDialog.getText(
            self,
            "保存导联组预设",
            "预设名称",
            text=suggested,
        )
        name = name.strip()
        if not accepted or not name:
            return
        if self._preset_store.get(name) is not None and QMessageBox.question(
            self,
            "覆盖预设",
            f"预设“{name}”已存在，是否覆盖？",
        ) != QMessageBox.Yes:
            return
        self._preset_store.save(name, self.groups())
        self._refresh_presets(name)

    def load_preset(self) -> None:
        if self._preset_store is None:
            return
        groups = self._preset_store.get(self.preset_combo.currentText())
        if groups is None:
            return
        available = set(self._available_channels)
        missing = list(dict.fromkeys(
            channel
            for group in groups
            for channel in group
            if channel not in available
        ))
        self._groups = [
            [channel for channel in group if channel in available]
            for group in groups
        ]
        self._groups = [group for group in self._groups if group]
        self._refresh_groups(0 if self._groups else -1)
        if missing:
            QMessageBox.information(
                self,
                "预设已按当前 EEG 调整",
                "当前 EEG 不包含以下导联，本次已忽略："
                + ", ".join(missing),
            )

    def delete_preset(self) -> None:
        if self._preset_store is None:
            return
        name = self.preset_combo.currentText()
        if not name:
            return
        if QMessageBox.question(
            self,
            "删除预设",
            f"确定删除导联组预设“{name}”？",
        ) != QMessageBox.Yes:
            return
        self._preset_store.delete(name)

    def _refresh_presets(self, selected: str | None = None) -> None:
        names = (
            sorted(self._preset_store.presets(), key=str.casefold)
            if self._preset_store is not None
            else []
        )
        selected = selected or self.preset_combo.currentText()
        with QSignalBlocker(self.preset_combo):
            self.preset_combo.clear()
            self.preset_combo.addItems(names)
            if selected in names:
                self.preset_combo.setCurrentText(selected)
        self._sync_preset_actions()

    def _sync_preset_actions(self, _index: int = -1) -> None:
        has_store = self._preset_store is not None
        has_preset = has_store and self.preset_combo.count() > 0
        self.preset_combo.setEnabled(has_store)
        self.save_preset_button.setEnabled(has_store)
        self.load_preset_button.setEnabled(has_preset)
        self.delete_preset_button.setEnabled(has_preset)

    def _refresh_groups(self, selected: int) -> None:
        with QSignalBlocker(self.group_list):
            self.group_list.clear()
            for index, group in enumerate(self._groups, start=1):
                summary = ", ".join(group) if group else "待添加导联"
                self.group_list.addItem(f"G{index} · {summary}")
            self.group_list.setCurrentRow(selected)
        self._group_changed(selected)

    def _group_changed(self, index: int) -> None:
        with QSignalBlocker(self.channel_list):
            self.channel_list.clear()
            if 0 <= index < len(self._groups):
                self.channel_list.addItems(self._groups[index])
        self._refresh_channel_choices(index)
        self._sync_actions()

    def _refresh_channel_choices(self, group_index: int) -> None:
        current = (
            set(self._groups[group_index])
            if 0 <= group_index < len(self._groups)
            else set()
        )
        with QSignalBlocker(self.channel_combo):
            self.channel_combo.clear()
            self.channel_combo.addItems(
                [channel for channel in self._available_channels if channel not in current]
            )

    def _sync_actions(self, _row: int = -1) -> None:
        has_group = self.group_list.currentRow() >= 0
        self.remove_group_button.setEnabled(has_group)
        self.channel_combo.setEnabled(has_group and self.channel_combo.count() > 0)
        self.add_channel_button.setEnabled(
            has_group and self.channel_combo.count() > 0
        )
        self.remove_channel_button.setEnabled(
            has_group and self.channel_list.currentRow() >= 0
        )

    def _accept_valid(self) -> None:
        try:
            ProcessingSpec(
                method=ProcessingMethod.WIENER,
                channel_groups=self.groups(),
            ).validate()
        except ValueError as exc:
            QMessageBox.warning(self, "导联组需要调整", str(exc))
            return
        self.accept()
