param([string]$LogDir = "", [int]$Port = 8008)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $LogDir) { $LogDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-qlora-v1\runs" }
tensorboard --logdir $LogDir --host 127.0.0.1 --port $Port
