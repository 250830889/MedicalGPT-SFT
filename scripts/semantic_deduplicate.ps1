param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,
    [string]$OutputDir = "",
    [string]$ConfigFile = "",
    [double]$Threshold = 0.92,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "data\deduplicated"
}
if (-not $ConfigFile) {
    $ConfigFile = Join-Path $Root "configs\bge-semantic-dedup.json"
}
if (-not (Test-Path $InputFile)) {
    throw "Input file not found: $InputFile"
}

python (Join-Path $Root "tools\semantic_deduplicate.py") `
  --input $InputFile `
  --output-dir $OutputDir `
  --config $ConfigFile `
  --similarity-threshold $Threshold `
  --device $Device
