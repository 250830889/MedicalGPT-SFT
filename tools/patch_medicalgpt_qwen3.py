#!/usr/bin/env python3
"""Patch MedicalGPT 2.7.0's qwen3_nothink template to match Qwen3 hard-switch formatting."""

from __future__ import annotations

import argparse
from pathlib import Path

ORIGINAL_PROMPT = 'prompt="<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n",'
PATCHED_PROMPT = (
    'prompt="<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n'
    '<think>\\n\\n</think>\\n\\n",'
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    args = parser.parse_args()

    template_path = args.upstream_dir / "training" / "template.py"
    if not template_path.exists():
        raise FileNotFoundError(f"MedicalGPT template file not found: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    marker = 'name="qwen3_nothink"'
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise RuntimeError("qwen3_nothink was not found. Confirm that MedicalGPT 2.7.0 is checked out.")

    block_start = text.rfind("register_conv_template(", 0, marker_pos)
    block_end = text.find("\n)\n", marker_pos)
    if block_start < 0 or block_end < 0:
        raise RuntimeError("Unable to isolate the qwen3_nothink template block.")

    block = text[block_start:block_end]
    if PATCHED_PROMPT in block:
        print(f"Qwen3 no-think template is already patched: {template_path}")
        return 0
    if ORIGINAL_PROMPT not in block:
        raise RuntimeError("The pinned template differs from the expected MedicalGPT 2.7.0 source; patch aborted.")

    patched_block = block.replace(ORIGINAL_PROMPT, PATCHED_PROMPT, 1)
    patched_text = text[:block_start] + patched_block + text[block_end:]
    template_path.write_text(patched_text, encoding="utf-8", newline="\n")
    print(f"Patched Qwen3 hard non-thinking prompt: {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
