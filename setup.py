from setuptools import setup, find_packages

setup(
    name="eeg_bg",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pywavelets>=1.4",
    ],
    extras_require={
        "gui": [
            "PySide6>=6.8,<7",
            "pyqtgraph>=0.13.7,<0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "eeg-bg-studio=eeg_bg.gui.app:main",
        ],
    },
)
