$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

python (Join-Path $Root "tools\prepare_dataset.py") `
  --input (Join-Path $Root "data\sample\medical_sft_semantic_duplicates.jsonl") `
  --format sharegpt `
  --output-dir (Join-Path $Root "data\processed-bge-sample") `
  --eval-ratio 0.34 `
  --seed 42 `
  --semantic-dedup `
  --dedup-model "BAAI/bge-small-zh-v1.5" `
  --dedup-threshold 0.92 `
  --dedup-device auto
