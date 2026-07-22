# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPECPATH).resolve().parents[1]
icon_path = ROOT / "packaging" / "windows" / "eeg_bg_studio.ico"

datas = collect_data_files("mne")
datas += [(str(ROOT / "configs" / "default.yaml"), "configs")]
datas += [(str(ROOT / "configs" / "erp_core_flankers.yaml"), "configs")]
hiddenimports = collect_submodules("mne", on_error="ignore")
hiddenimports += [
    "edfio",
    "sklearn.decomposition",
    "sklearn.utils._cython_blas",
    "scipy.signal",
]

a = Analysis(
    [str(ROOT / "eeg_bg_studio.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch", "xgboost", "shap", "jupyter", "IPython",
        "pandas", "pyarrow", "sqlalchemy", "numba", "llvmlite",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="eeg_bg_studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_path) if icon_path.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="eeg_bg_studio",
)
