param(
    [string]$BaseModel = "Qwen/Qwen3-8B",
    [string]$AdapterDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Upstream = Join-Path $Root "vendor\MedicalGPT"

if (-not $AdapterDir) { $AdapterDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-qlora-v1" }
if (-not $OutputDir) { $OutputDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-merged-v1" }

if (-not (Test-Path $Upstream)) { throw "Run scripts/setup_upstream.ps1 first." }
if (-not (Test-Path $AdapterDir)) { throw "LoRA adapter not found: $AdapterDir" }

$MergeArgs = @(
    "--base_model", $BaseModel,
    "--tokenizer_path", $BaseModel,
    "--lora_model", $AdapterDir,
    "--output_dir", $OutputDir
)

Push-Location $Upstream
try {
    & python "tools/merge_peft_adapter.py" @MergeArgs
    if ($LASTEXITCODE -ne 0) { throw "LoRA merge failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
