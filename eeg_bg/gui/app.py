from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .branding import APPLICATION_NAME, SETTINGS_ORGANIZATION
from .main_window import MainWindow
from .theme import stylesheet


def create_application(argv: list[str] | None = None) -> QApplication:
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(stylesheet())
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=APPLICATION_NAME)
    parser.add_argument("--smoke-test", action="store_true")
    args, qt_args = parser.parse_known_args(argv)
    if args.smoke_test:
        from eeg_bg.gui.smoke import run_smoke_test

        return run_smoke_test()
    app = create_application([sys.argv[0], *qt_args])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
