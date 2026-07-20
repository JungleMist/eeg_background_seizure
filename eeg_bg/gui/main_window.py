from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .pages import BatchPage, PreviewPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("eeg_bg Studio")
        self.resize(1480, 900)
        self.setMinimumSize(1120, 700)

        root = QWidget()
        root.setObjectName("AppRoot")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("Navigation")
        nav.setFixedWidth(184)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(18, 22, 18, 18)
        nav_layout.setSpacing(8)
        brand = QLabel("eeg_bg")
        brand.setObjectName("Brand")
        accent = QLabel("SIGNAL STUDIO")
        accent.setObjectName("BrandAccent")
        nav_layout.addWidget(brand)
        nav_layout.addWidget(accent)
        nav_layout.addSpacing(24)
        self.preview_nav = QPushButton("⌁  交互预览")
        self.batch_nav = QPushButton("▦  批量处理")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for button in (self.preview_nav, self.batch_nav):
            button.setObjectName("NavButton")
            button.setCheckable(True)
            group.addButton(button)
            nav_layout.addWidget(button)
        self.preview_nav.setChecked(True)
        nav_layout.addStretch(1)
        version = QLabel("EEG denoising\nworkspace · v0.1.0")
        version.setObjectName("Muted")
        version.setWordWrap(True)
        nav_layout.addWidget(version)
        layout.addWidget(nav)

        self.stack = QStackedWidget()
        self.preview_page = PreviewPage()
        self.batch_page = BatchPage()
        self.stack.addWidget(self.preview_page)
        self.stack.addWidget(self.batch_page)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.preview_nav.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.batch_nav.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.preview_page.statusChanged.connect(self._show_status)
        self.batch_page.statusChanged.connect(self._show_status)
        status = QStatusBar()
        status.setSizeGripEnabled(True)
        self.setStatusBar(status)
        self._show_status("就绪 · 打开一个 EDF 或 FIF 文件开始")

        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.preview_page.choose_file)
        self.addAction(open_action)
        process_action = QAction(self)
        process_action.setShortcut(QKeySequence("Ctrl+Return"))
        process_action.triggered.connect(self.preview_page.process)
        self.addAction(process_action)

    def _show_status(self, text: str):
        self.statusBar().showMessage(text)

    def closeEvent(self, event):
        active = self.preview_page._thread is not None or self.batch_page._thread is not None
        if active:
            self.preview_page.request_cancel()
            self.batch_page.request_cancel()
            self._show_status("已请求安全取消；任务停止后可关闭窗口")
            event.ignore()
            return
        event.accept()
