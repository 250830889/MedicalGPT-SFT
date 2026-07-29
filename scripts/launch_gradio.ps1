param(
    [string]$ModelDir = "",
    [string]$AdapterDir = "",
    [int]$Port = 8081,
    [switch]$Thinking,
    [switch]$NoQuantization
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $ModelDir) { $ModelDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-merged-v1" }

$ArgsList = @(
    (Join-Path $Root "tools\qwen3_gradio.py"),
    "--model", $ModelDir,
    "--port", "$Port",
    "--max-new-tokens", "512",
    "--temperature", "0.7",
    "--top-p", "0.8",
    "--top-k", "20"
)

if ($AdapterDir) { $ArgsList += @("--adapter", $AdapterDir) }
if ($Thinking) { $ArgsList += "--thinking" }
if (-not $NoQuantization) { $ArgsList += "--load-in-4bit" }

& python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Gradio launch failed with exit code $LASTEXITCODE." }
