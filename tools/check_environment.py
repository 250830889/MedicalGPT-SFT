#!/usr/bin/env python3
"""Print runtime compatibility information for Qwen3-8B QLoRA training."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes", "gradio", "sentence-transformers", "numpy")
        },
    }

    try:
        import torch

        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "name": torch.cuda.get_device_name(index),
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            })
        result.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": gpus,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        })
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from transformers import AutoConfig
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: F401
        result["qwen3_architecture_available"] = True
        result["qwen3_config_class"] = AutoConfig.for_model("qwen3").__class__.__name__
    except Exception as exc:
        result["qwen3_architecture_available"] = False
        result["qwen3_error"] = f"{type(exc).__name__}: {exc}"

    warnings: list[str] = []
    if not result.get("cuda_available"):
        warnings.append("CUDA GPU not detected; Qwen3-8B QLoRA training cannot run in this environment.")
    if result["packages"].get("bitsandbytes") is None:
        warnings.append("bitsandbytes is missing; 4-bit QLoRA will not work.")
    if result["packages"].get("sentence-transformers") is None:
        warnings.append("sentence-transformers is missing; BGE semantic deduplication will not work.")
    if not result.get("qwen3_architecture_available"):
        warnings.append("Transformers does not expose Qwen3; reinstall the pinned upstream requirements.")
    result["warnings"] = warnings

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if not result.get("qwen3_architecture_available") else 0


if __name__ == "__main__":
    raise SystemExit(main())
