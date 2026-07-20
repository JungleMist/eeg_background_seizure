from __future__ import annotations


COLORS = {
    "canvas": "#0B1118",
    "surface": "#111B24",
    "raised": "#172431",
    "border": "#263746",
    "text": "#E6EEF5",
    "muted": "#8FA4B7",
    "raw": "#21B8A6",
    "processed": "#4B8DFF",
    "warning": "#F4B860",
    "danger": "#FF6B6B",
    "success": "#52C98B",
}


def stylesheet() -> str:
    c = COLORS
    return f"""
    * {{
        font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        font-size: 13px;
        color: {c['text']};
    }}
    QMainWindow, QWidget#AppRoot {{ background: {c['canvas']}; }}
    QWidget#Navigation {{
        background: {c['surface']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#Brand {{ font-size: 20px; font-weight: 700; color: {c['text']}; }}
    QLabel#BrandAccent {{ font-size: 11px; color: {c['raw']}; letter-spacing: 1px; }}
    QLabel#PageTitle {{ font-size: 22px; font-weight: 650; }}
    QLabel#SectionTitle {{ font-size: 14px; font-weight: 650; }}
    QLabel#Muted, QLabel#Metadata {{ color: {c['muted']}; }}
    QLabel#Metadata {{ font-family: "Cascadia Mono", "Consolas", monospace; }}
    QFrame#Panel, QGroupBox {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 7px;
    }}
    QGroupBox {{ margin-top: 13px; padding: 12px 10px 10px; font-weight: 600; }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {c['muted']};
    }}
    QPushButton, QToolButton {{
        min-height: 34px;
        padding: 0 12px;
        background: {c['raised']};
        border: 1px solid {c['border']};
        border-radius: 6px;
    }}
    QPushButton:hover, QToolButton:hover {{ border-color: {c['raw']}; }}
    QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {c['processed']}; }}
    QPushButton:disabled, QToolButton:disabled {{ color: #607182; background: #121B23; }}
    QPushButton#Primary {{
        background: {c['processed']};
        border-color: {c['processed']};
        color: white;
        font-weight: 650;
    }}
    QPushButton#Danger {{ color: {c['danger']}; }}
    QPushButton#NavButton {{
        min-height: 42px;
        text-align: left;
        padding-left: 14px;
        border-color: transparent;
        background: transparent;
        color: {c['muted']};
    }}
    QPushButton#NavButton:checked {{
        color: {c['text']};
        background: {c['raised']};
        border-left: 3px solid {c['raw']};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        min-height: 32px;
        padding: 0 8px;
        background: #0E171F;
        border: 1px solid {c['border']};
        border-radius: 5px;
        selection-background-color: {c['processed']};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QCheckBox {{ spacing: 8px; }}
    QProgressBar {{
        min-height: 9px; max-height: 9px;
        background: #0E171F; border: none; border-radius: 4px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {c['raw']}; border-radius: 4px; }}
    QTableWidget {{
        background: {c['surface']}; alternate-background-color: #101923;
        border: 1px solid {c['border']}; border-radius: 7px;
        gridline-color: {c['border']}; selection-background-color: #1D3A52;
    }}
    QHeaderView::section {{
        background: {c['raised']}; color: {c['muted']}; border: none;
        border-bottom: 1px solid {c['border']}; padding: 8px;
    }}
    QScrollArea, QScrollArea > QWidget, QWidget#ScrollContents {{
        border: none; background: {c['canvas']};
    }}
    QScrollBar:vertical {{
        background: {c['canvas']}; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']}; min-height: 28px; border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: {c['canvas']}; height: 10px; margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['border']}; min-width: 28px; border-radius: 5px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QSplitter::handle {{ background: {c['border']}; width: 1px; height: 1px; }}
    QStatusBar {{ background: {c['surface']}; border-top: 1px solid {c['border']}; }}
    QToolTip {{ background: {c['raised']}; border: 1px solid {c['border']}; }}
    """
