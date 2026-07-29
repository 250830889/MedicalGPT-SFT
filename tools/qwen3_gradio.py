#!/usr/bin/env python3
"""Gradio chat UI for a merged Qwen3 model or a base model plus LoRA adapter."""

from __future__ import annotations

import argparse

import gradio as gr

from qwen3_runtime import DEFAULT_SYSTEM_PROMPT, final_answer_only, load_model_and_tokenizer, stream_answer


def history_to_messages(history) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": final_answer_only(str(content))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                messages.append({"role": "assistant", "content": final_answer_only(str(assistant_text))})
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter, args.load_in_4bit)

    def respond(message, history):
        messages = [
            {"role": "system", "content": args.system_prompt},
            *history_to_messages(history),
            {"role": "user", "content": message},
        ]
        partial = ""
        for chunk in stream_answer(
            model,
            tokenizer,
            messages,
            thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ):
            partial += chunk
            yield partial

    mode = "思考模式" if args.thinking else "非思考模式"
    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="MedicalGPT · Qwen3-8B",
        description=f"单卡 QLoRA 医疗问答演示（{mode}）。输出仅供研究与信息参考，不能替代专业诊疗。",
        examples=["乙肝和丙肝有哪些主要区别？", "持续高热并伴有呼吸困难时应该怎么办？"],
    )
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
