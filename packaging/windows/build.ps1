$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $RepoRoot ".venv-eeg-bg-studio"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -3.11 -m venv $Venv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $RepoRoot "requirements-gui.txt")
& $Python -m pip install -e $RepoRoot
& $Python (Join-Path $PSScriptRoot "make_icon.py")

Push-Location $RepoRoot
try {
    & $Python -m pytest tests/test_application tests/test_gui -q
    & $Python -m eeg_bg.gui --smoke-test
    & $Python -m PyInstaller --clean --noconfirm (Join-Path $PSScriptRoot "eeg_bg_studio.spec")
    $WarningFile = Join-Path $RepoRoot "build\eeg_bg_studio\warn-eeg_bg_studio.txt"
    if (-not (Test-Path $WarningFile)) {
        throw "PyInstaller warning file not found: $WarningFile"
    }
    $CriticalWarnings = Select-String -Path $WarningFile -Pattern `
        "missing module named '(eeg_bg|PySide6|pyqtgraph|mne|sklearn|scipy|edfio)(\.|')"
    if ($CriticalWarnings) {
        $CriticalWarnings | ForEach-Object { Write-Error $_.Line }
        throw "PyInstaller reported missing application runtime modules."
    }
    & (Join-Path $PSScriptRoot "smoke_test.ps1") -DistRoot (Join-Path $RepoRoot "dist\eeg_bg_studio")
    $Archive = Join-Path $RepoRoot "dist\eeg_bg_studio-windows-x64.zip"
    if (Test-Path $Archive) { Remove-Item $Archive -Force }
    Compress-Archive -Path (Join-Path $RepoRoot "dist\eeg_bg_studio\*") -DestinationPath $Archive
    Write-Host "Build complete: $Archive"
}
finally {
    Pop-Location
}
