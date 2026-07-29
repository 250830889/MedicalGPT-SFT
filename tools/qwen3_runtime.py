#!/usr/bin/env python3
"""Shared loading and generation helpers for Qwen3 medical demos."""

from __future__ import annotations

import re
from threading import Thread
from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

DEFAULT_SYSTEM_PROMPT = (
    "你是一个医疗领域问答助手。请基于可靠医学知识清晰回答；信息不足时明确说明不确定性。"
    "对于急症、危险症状或需要诊断和处方的情况，提醒用户及时咨询合格医疗专业人员。"
)


def load_model_and_tokenizer(model_name_or_path: str, adapter: str = "", load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, object] = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit bitsandbytes inference requires a compatible CUDA GPU.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, device_map="auto")
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, messages: list[dict[str, str]], thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        # Compatibility fallback for tokenizers that do not expose enable_thinking.
        fallback = [dict(message) for message in messages]
        if not thinking:
            if fallback and fallback[0].get("role") == "system":
                fallback[0]["content"] = f"{fallback[0]['content']}\n/no_think"
            else:
                fallback.insert(0, {"role": "system", "content": "/no_think"})
        return tokenizer.apply_chat_template(fallback, tokenize=False, add_generation_prompt=True)


def stream_answer(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterable[str]:
    prompt = build_prompt(tokenizer, messages, thinking)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    yield from streamer
    thread.join()


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def final_answer_only(text: str) -> str:
    """Remove reasoning blocks before adding assistant output back to chat history."""
    return _THINK_BLOCK.sub("", text).strip()
