# MedicalGPT · Qwen3-8B Medical QLoRA SFT

医疗领域大模型监督微调项目。当前仓库已完成对 `Qwen/Qwen3-8B` 的单卡 4-bit QLoRA 配置、BGE 语义去重、数据处理、训练、Adapter 合并、CLI/Gradio 推理和实验记录规范化。

## 项目亮点

- **Qwen3-8B 单卡 QLoRA：** 使用 4-bit NF4、double quantization、梯度检查点和梯度累积，将 8B 模型微调配置适配到常见单卡环境。
- **训练/推理模板一致：** 固定 MedicalGPT `2.7.0`，并对其 `qwen3_nothink` 模板执行幂等补丁，使训练 Prompt 与 Qwen3 `enable_thinking=False` 的硬开关格式一致。
- **完整数据工程：** 支持 QA、Alpaca、ShareGPT 三类数据转换，执行格式校验、精确去重、BGE 向量化余弦相似度去重及确定性 train/eval 划分，降低语义近似样本跨集合泄漏。
- **完整工程闭环：** 提供环境检查、训练、Adapter 合并、TensorBoard、CLI 和 Gradio 脚本；本地推理支持思考/非思考模式切换。

## 默认配置

| 项目 | 配置 |
|---|---|
| Base model | `Qwen/Qwen3-8B` |
| Upstream | MedicalGPT `2.7.0` |
| Method | Single-GPU 4-bit QLoRA SFT |
| Template | `qwen3_nothink` |
| Context | 2048 Token |
| LoRA | Rank 16 / Alpha 32 / Dropout 0.1 |
| Target modules | All linear layers |
| Micro batch | 1 |
| Gradient accumulation | 16 |
| Effective batch | 16 |
| Epochs | 2 |
| Learning rate | `2e-5` |

完整参数见 `configs/qwen3-8b-qlora-sft.json`。

## 目录结构

```text
.
├── configs/                 # Qwen3 与 BGE 语义去重配置
├── data/sample/             # 可提交到 GitHub 的小型数据样例
├── docs/                    # Qwen3-8B 配置和复现说明
├── results/                 # 不含权重的实验指标与结果模板
├── scripts/                 # Windows PowerShell 执行脚本
├── tests/                   # BGE 去重核心逻辑测试
└── tools/                   # 数据、BGE 去重、模板补丁及 Qwen3 推理工具
```

## 快速开始

### 1. 安装并固定 MedicalGPT

```powershell
.\scripts\setup_upstream.ps1
```

### 2. 准备数据

验证样例链路：

```powershell
.\scripts\prepare_sample_data.ps1
```

处理自己的 ShareGPT 数据：

```powershell
python .\tools\prepare_dataset.py `
  --input .\data\raw\medical.jsonl `
  --format sharegpt `
  --output-dir .\data\processed `
  --eval-ratio 0.1 `
  --seed 42
```

`--format` 还支持 `qa` 和 `alpaca`。

启用 BGE 语义去重，并在去重后再划分训练集和验证集：

```powershell
python .\tools\prepare_dataset.py `
  --input .\data\raw\medical.jsonl `
  --format sharegpt `
  --output-dir .\data\processed `
  --eval-ratio 0.1 `
  --seed 42 `
  --semantic-dedup `
  --dedup-model BAAI/bge-small-zh-v1.5 `
  --dedup-threshold 0.92
```

也可以对规范化后的 ShareGPT 数据单独运行：

```powershell
.\scripts\semantic_deduplicate.ps1 `
  -InputFile ".\data\normalized\medical.jsonl" `
  -Threshold 0.92
```

### 3. 运行 Qwen3-8B QLoRA SFT

```powershell
.\scripts\train_sft_windows.ps1
```

```powershell
.\scripts\train_sft_windows.ps1 -ModelMaxLength 1024 -MaxTrainSamples 100
```

脚本会自动检测 BF16 支持；不支持时回退到 FP16 计算。默认使用全部已处理数据，`MaxTrainSamples=-1`、`MaxEvalSamples=-1`。

### 4. 合并 Adapter

```powershell
.\scripts\merge_lora.ps1
```

默认输出：

```text
outputs-sft-qwen3-8b-medical-merged-v1
```

### 5. CLI 和 Gradio 推理

```powershell
.\scripts\inference_cli.ps1
.\scripts\launch_gradio.ps1
```

两者默认以 4-bit 加载合并模型并关闭思考输出。启用思考模式：

```powershell
.\scripts\inference_cli.ps1 -Thinking
.\scripts\launch_gradio.ps1 -Thinking
```

直接加载基座模型和未合并 Adapter：

```powershell
.\scripts\inference_cli.ps1 `
  -ModelDir "Qwen/Qwen3-8B" `
  -AdapterDir ".\outputs-sft-qwen3-8b-medical-qlora-v1"
```

### 6. TensorBoard

```powershell
.\scripts\launch_tensorboard.ps1
```

## License

本仓库保留 Apache License 2.0。
