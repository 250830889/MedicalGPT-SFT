#!/usr/bin/env python3
"""Convert QA/Alpaca/ShareGPT JSON or JSONL data to validated ShareGPT JSONL.

The tool also performs deterministic train/eval splitting so the two datasets do
not accidentally point to the same source directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from semantic_deduplicate import SemanticDedupConfig, deduplicate_records, write_jsonl as write_dedup_jsonl

VALID_USER_ROLES = {"human", "user"}
VALID_ASSISTANT_ROLES = {"gpt", "assistant"}


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Input file is empty")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            records.append(item)
        return records
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise ValueError("JSON input must be an object or an array of objects")


def normalize_message(role: str, content: Any) -> dict[str, str]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Conversation message content must be a non-empty string")
    role_norm = str(role).strip().lower()
    if role_norm in VALID_USER_ROLES:
        output_role = "human"
    elif role_norm in VALID_ASSISTANT_ROLES:
        output_role = "gpt"
    elif role_norm == "system":
        output_role = "system"
    else:
        raise ValueError(f"Unsupported role: {role}")
    return {"from": output_role, "value": content.strip()}


def to_sharegpt(record: dict[str, Any], data_format: str) -> dict[str, Any]:
    if data_format == "qa":
        question = record.get("question", record.get("input"))
        answer = record.get("answer", record.get("output"))
        messages = [normalize_message("human", question), normalize_message("gpt", answer)]
    elif data_format == "alpaca":
        instruction = record.get("instruction")
        extra_input = record.get("input", "")
        output = record.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Alpaca record is missing a non-empty instruction")
        prompt = instruction.strip()
        if isinstance(extra_input, str) and extra_input.strip():
            prompt = f"{prompt}\n\n{extra_input.strip()}"
        messages = [normalize_message("human", prompt), normalize_message("gpt", output)]
    else:
        raw_messages = record.get("conversations", record.get("messages"))
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("ShareGPT record is missing conversations/messages")
        messages = []
        for message in raw_messages:
            if not isinstance(message, dict):
                raise ValueError("Each conversation entry must be an object")
            role = message.get("from", message.get("role"))
            content = message.get("value", message.get("content"))
            messages.append(normalize_message(role, content))

    dialogue_roles = [m["from"] for m in messages if m["from"] != "system"]
    if len(dialogue_roles) < 2 or dialogue_roles[0] != "human" or dialogue_roles[-1] != "gpt":
        raise ValueError("Conversation must start with a user turn and end with an assistant turn")
    for left, right in zip(dialogue_roles, dialogue_roles[1:]):
        if left == right:
            raise ValueError("User and assistant turns must alternate")
    return {"conversations": messages}


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(record)
    return output, removed


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--format", required=True, choices=("qa", "alpaca", "sharegpt"))
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--eval-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--semantic-dedup",
        action="store_true",
        help="Run BGE cosine-similarity deduplication before train/eval splitting",
    )
    parser.add_argument("--dedup-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--dedup-threshold", default=0.92, type=float)
    parser.add_argument(
        "--dedup-text-mode",
        default="first_user",
        choices=("first_user", "all_user", "full_dialogue"),
    )
    parser.add_argument("--dedup-batch-size", default=64, type=int)
    parser.add_argument("--dedup-compare-chunk-size", default=512, type=int)
    parser.add_argument("--dedup-device", default="auto")
    parser.add_argument("--dedup-max-seq-length", default=512, type=int)
    args = parser.parse_args()

    if not 0 < args.eval_ratio < 1:
        parser.error("--eval-ratio must be between 0 and 1")

    raw_records = read_records(args.input)
    converted: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(raw_records, start=1):
        try:
            converted.append(to_sharegpt(record, args.format))
        except (TypeError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")

    converted, exact_duplicates_removed = deduplicate(converted)
    semantic_report: dict[str, Any] | None = None
    semantic_removed: list[dict[str, Any]] = []
    if args.semantic_dedup:
        semantic_config = SemanticDedupConfig(
            model_name_or_path=args.dedup_model,
            similarity_threshold=args.dedup_threshold,
            text_mode=args.dedup_text_mode,
            batch_size=args.dedup_batch_size,
            compare_chunk_size=args.dedup_compare_chunk_size,
            device=args.dedup_device,
            max_seq_length=args.dedup_max_seq_length,
        )
        converted, semantic_report, semantic_removed = deduplicate_records(
            converted,
            semantic_config,
            perform_exact_dedup=False,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "semantic_dedup_report.json").write_text(
            json.dumps(semantic_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_dedup_jsonl(
            args.output_dir / "semantic_duplicates_removed.jsonl",
            semantic_removed,
        )

    if len(converted) < 2:
        raise ValueError("At least two valid, unique records are required for splitting")

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    eval_count = max(1, round(len(converted) * args.eval_ratio))
    eval_records = converted[:eval_count]
    train_records = converted[eval_count:]
    if not train_records:
        raise ValueError("Evaluation split consumed all records")

    write_jsonl(args.output_dir / "train" / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "eval" / "eval.jsonl", eval_records)
    stats = {
        "input_records": len(raw_records),
        "valid_unique_records": len(converted),
        "exact_duplicates_removed": exact_duplicates_removed,
        "semantic_dedup_enabled": args.semantic_dedup,
        "semantic_duplicates_removed": (semantic_report or {}).get("semantic_duplicates_removed", 0),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "invalid_records": len(errors),
        "seed": args.seed,
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if errors:
        print("Skipped invalid records:")
        for error in errors[:20]:
            print(f"- {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
