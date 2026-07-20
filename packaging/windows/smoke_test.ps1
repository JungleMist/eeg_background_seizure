param(
    [Parameter(Mandatory = $true)]
    [string]$DistRoot
)

$ErrorActionPreference = "Stop"
$Exe = Join-Path $DistRoot "eeg_bg_studio.exe"
if (-not (Test-Path $Exe)) {
    throw "Packaged executable not found: $Exe"
}

$Process = Start-Process -FilePath $Exe -ArgumentList "--smoke-test" -Wait -PassThru
if ($Process.ExitCode -ne 0) {
    throw "Packaged smoke test failed with exit code $($Process.ExitCode)"
}
Write-Host "Packaged smoke test passed."
