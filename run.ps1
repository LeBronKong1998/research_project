param(
    [ValidateSet('train', 'evaluate', 'smoke', 'download')][string]$Command = 'smoke',
    [string]$Checkpoint = 'runs/interaction-real/best.pt',
    [string]$DataDir = 'data/interaction'
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    if ($Command -eq 'train') { & $Python -m vehtraj train --data-dir $DataDir --epochs 30 --batch-size 32 --accumulation 1 --lr 0.001 --output runs/interaction-real }
    elseif ($Command -eq 'evaluate') { & $Python -m vehtraj evaluate --data-dir $DataDir --checkpoint $Checkpoint --output runs/interaction-real }
    else { & $Python -m vehtraj $Command }
    if ($LASTEXITCODE -ne 0) { throw "$Command failed." }
} finally { Pop-Location }
