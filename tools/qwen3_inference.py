#!/usr/bin/env python3
"""Interactive Qwen3 CLI with an explicit thinking/non-thinking switch."""

from __future__ import annotations

import argparse

from qwen3_runtime import DEFAULT_SYSTEM_PROMPT, final_answer_only, load_model_and_tokenizer, stream_answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged model path or Hugging Face model ID")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter path")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 reasoning output")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter, args.load_in_4bit)
    history: list[dict[str, str]] = []
    print("Qwen3 Medical CLI. Commands: clear, exit")
    print(f"Thinking mode: {'enabled' if args.thinking else 'disabled'}")

    while True:
        try:
            query = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "clear":
            history.clear()
            print("History cleared.")
            continue

        messages = [{"role": "system", "content": args.system_prompt}, *history, {"role": "user", "content": query}]
        print("Assistant: ", end="", flush=True)
        chunks: list[str] = []
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
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()
        response = "".join(chunks)
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": final_answer_only(response)},
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
