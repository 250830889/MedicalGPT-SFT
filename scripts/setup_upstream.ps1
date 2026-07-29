$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/shibing624/MedicalGPT.git"
$Revision = "2.7.0"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Target = Join-Path $Root "vendor\MedicalGPT"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git and reopen PowerShell."
}

if (-not (Test-Path $Target)) {
    New-Item -ItemType Directory -Force (Split-Path $Target) | Out-Null
    git clone $RepoUrl $Target
}

Push-Location $Target
try {
    git fetch --all --tags --force
    git checkout --detach $Revision
    git reset --hard $Revision

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt --upgrade
    python -m pip install -r (Join-Path $Root "requirements-extra.txt") --upgrade
} finally {
    Pop-Location
}

# MedicalGPT 2.7.0 defines qwen3_nothink, but its prompt is identical to the
# thinking template. Apply an idempotent local patch so training and inference
# use Qwen3's hard non-thinking prefix consistently.
python (Join-Path $Root "tools\patch_medicalgpt_qwen3.py") `
  --upstream-dir $Target

python (Join-Path $Root "tools\check_environment.py")
Write-Host "Pinned upstream MedicalGPT $Revision is ready at $Target"
