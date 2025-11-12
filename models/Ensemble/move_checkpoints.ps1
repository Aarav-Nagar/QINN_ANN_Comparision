# Helper PowerShell script to move ensemble checkpoint files into the package's checkpoints folder
$src = Join-Path $PSScriptRoot '..'\
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..' )
$files = Get-ChildItem -Path "$projectRoot\models" -Filter "ensemble_*_TSLA_*_chkpt.pth" -File -ErrorAction SilentlyContinue
if (-not (Test-Path "$PSScriptRoot\checkpoints")) { New-Item -ItemType Directory -Path "$PSScriptRoot\checkpoints" | Out-Null }
foreach ($f in $files) {
    try {
        Move-Item -Path $f.FullName -Destination (Join-Path $PSScriptRoot 'checkpoints') -Force
        Write-Host "Moved $($f.Name)"
    } catch {
        Write-Warning "Failed to move $($f.Name): $_"
    }
}
Write-Host "Done."
