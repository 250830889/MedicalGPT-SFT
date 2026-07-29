param(
    [string]$Model = "Qwen/Qwen3-8B",
    [string]$TrainDir = "",
    [string]$EvalDir = "",
    [string]$OutputDir = "",
    [int]$ModelMaxLength = 2048,
    [double]$Epochs = 2,
    [int]$GradientAccumulationSteps = 16,
    [int]$MaxTrainSamples = -1,
    [int]$MaxEvalSamples = -1
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Upstream = Join-Path $Root "vendor\MedicalGPT"

if (-not $TrainDir) { $TrainDir = Join-Path $Root "data\processed\train" }
if (-not $EvalDir) { $EvalDir = Join-Path $Root "data\processed\eval" }
if (-not $OutputDir) { $OutputDir = Join-Path $Root "outputs-sft-qwen3-8b-medical-qlora-v1" }

if (-not (Test-Path $Upstream)) { throw "Run scripts/setup_upstream.ps1 first." }
if (-not (Test-Path $TrainDir)) { throw "Training data not found: $TrainDir" }
if (-not (Test-Path $EvalDir)) { throw "Evaluation data not found: $EvalDir" }
if ($ModelMaxLength -lt 512) { throw "ModelMaxLength must be at least 512." }

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Prefer BF16 on supported GPUs; fall back to FP16 for older CUDA cards.
$UseBf16 = python -c "import torch; print('1' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect CUDA/BF16 support." }

$TrainArgs = @(
    "--model_name_or_path", $Model,
    "--train_file_dir", $TrainDir,
    "--validation_file_dir", $EvalDir,
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--do_train",
    "--do_eval",
    "--template_name", "qwen3_nothink",
    "--use_peft", "True",
    "--load_in_4bit", "True",
    "--qlora", "True",
    "--max_train_samples", "$MaxTrainSamples",
    "--max_eval_samples", "$MaxEvalSamples",
    "--model_max_length", "$ModelMaxLength",
    "--num_train_epochs", "$Epochs",
    "--learning_rate", "2e-5",
    "--warmup_ratio", "0.03",
    "--weight_decay", "0.05",
    "--logging_strategy", "steps",
    "--logging_steps", "10",
    "--eval_strategy", "steps",
    "--eval_steps", "50",
    "--save_strategy", "steps",
    "--save_steps", "100",
    "--save_total_limit", "3",
    "--gradient_accumulation_steps", "$GradientAccumulationSteps",
    "--preprocessing_num_workers", "0",
    "--overwrite_cache", "True",
    "--output_dir", $OutputDir,
    "--ddp_timeout", "30000",
    "--logging_first_step", "True",
    "--target_modules", "all",
    "--lora_rank", "16",
    "--lora_alpha", "32",
    "--lora_dropout", "0.1",
    "--report_to", "tensorboard",
    "--ddp_find_unused_parameters", "False",
    "--gradient_checkpointing", "True",
    "--cache_dir", (Join-Path $Root "cache")
)

if ($UseBf16.Trim() -eq "1") {
    $TrainArgs += @("--torch_dtype", "bfloat16", "--bf16")
    Write-Host "Precision: BF16 compute with 4-bit NF4 QLoRA"
} else {
    $TrainArgs += @("--torch_dtype", "float16", "--fp16")
    Write-Host "Precision: FP16 compute with 4-bit NF4 QLoRA"
}

Write-Host "Model: $Model"
Write-Host "Train data: $TrainDir"
Write-Host "Eval data: $EvalDir"
Write-Host "Output: $OutputDir"
Write-Host "Context length: $ModelMaxLength"

Push-Location $Upstream
try {
    & python "training/supervised_finetuning.py" @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "Qwen3-8B QLoRA training failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
