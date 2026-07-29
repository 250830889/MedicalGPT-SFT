$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
python (Join-Path $Root "tools\prepare_dataset.py") `
  --input (Join-Path $Root "data\sample\medical_sft_sample.jsonl") `
  --format sharegpt `
  --output-dir (Join-Path $Root "data\processed") `
  --eval-ratio 0.34 `
  --seed 42
