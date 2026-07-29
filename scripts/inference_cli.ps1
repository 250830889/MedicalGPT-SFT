param(
    [string]$ModelDir = "",
    [string]$AdapterDir = "",
    [switch]$Thinking,
    [switch]$NoQuantization
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $ModelDir) { $ModelDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-merged-v1" }

$ArgsList = @(
    (Join-Path $Root "tools\qwen3_inference.py"),
    "--model", $ModelDir,
    "--max-new-tokens", "512",
    "--temperature", "0.7",
    "--top-p", "0.8",
    "--top-k", "20"
)

if ($AdapterDir) { $ArgsList += @("--adapter", $AdapterDir) }
if ($Thinking) { $ArgsList += "--thinking" }
if (-not $NoQuantization) { $ArgsList += "--load-in-4bit" }

& python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "CLI inference failed with exit code $LASTEXITCODE." }
