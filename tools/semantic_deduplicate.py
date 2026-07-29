#!/usr/bin/env python3
"""BGE embedding + cosine-similarity semantic deduplication for ShareGPT data.

The script keeps the earliest example in each near-duplicate group, writes the
retained dataset, and records every removed example with its nearest earlier
match. Embeddings are L2-normalized, so the dot product used by the blockwise
matcher is exactly cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_THRESHOLD = 0.92
VALID_TEXT_MODES = ("first_user", "all_user", "full_dialogue")


@dataclass(frozen=True)
class SemanticDedupConfig:
    model_name_or_path: str = DEFAULT_MODEL
    similarity_threshold: float = DEFAULT_THRESHOLD
    text_mode: str = "first_user"
    batch_size: int = 64
    compare_chunk_size: int = 512
    device: str = "auto"
    max_seq_length: int = 512
    trust_remote_code: bool = False

    def validate(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.text_mode not in VALID_TEXT_MODES:
            raise ValueError(f"text_mode must be one of {VALID_TEXT_MODES}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.compare_chunk_size <= 0:
            raise ValueError("compare_chunk_size must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Input file is empty")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def exact_deduplicate(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove byte-equivalent JSON records while preserving input order."""
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for source_index, record in enumerate(records):
        key = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key in seen:
            removed.append(
                {
                    "duplicate_source_index": source_index,
                    "retained_source_index": seen[key],
                    "reason": "exact_duplicate",
                    "similarity": 1.0,
                    "record": record,
                }
            )
            continue
        seen[key] = source_index
        retained.append(record)
    return retained, removed


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("conversations", record.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("Record is missing a non-empty conversations/messages list")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("Every conversation entry must be an object")
    return messages


def _role_and_content(message: dict[str, Any]) -> tuple[str, str]:
    role = str(message.get("from", message.get("role", ""))).strip().lower()
    content = message.get("value", message.get("content"))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Conversation message content must be a non-empty string")
    return role, content.strip()


def extract_semantic_text(record: dict[str, Any], mode: str = "first_user") -> str:
    """Build the text that represents one training sample for semantic matching."""
    messages = _messages(record)
    parsed = [_role_and_content(message) for message in messages]
    user_roles = {"human", "user"}

    if mode == "first_user":
        for role, content in parsed:
            if role in user_roles:
                return content
        raise ValueError("Record has no user/human turn")

    if mode == "all_user":
        user_turns = [content for role, content in parsed if role in user_roles]
        if not user_turns:
            raise ValueError("Record has no user/human turn")
        return "\n".join(user_turns)

    if mode == "full_dialogue":
        return "\n".join(f"{role}: {content}" for role, content in parsed)

    raise ValueError(f"Unsupported text mode: {mode}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def encode_with_bge(texts: Sequence[str], config: SemanticDedupConfig) -> "Any":
    """Encode texts with a SentenceTransformers-compatible BGE model."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE semantic deduplication requires sentence-transformers. "
            "Run: python -m pip install -r requirements-extra.txt"
        ) from exc

    device = resolve_device(config.device)
    model = SentenceTransformer(
        config.model_name_or_path,
        device=device,
        trust_remote_code=config.trust_remote_code,
    )
    model.max_seq_length = config.max_seq_length
    embeddings = model.encode(
        list(texts),
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise RuntimeError(f"Unexpected embedding shape: {array.shape}")
    return array


def cosine_greedy_deduplicate(
    embeddings: "Any",
    threshold: float,
    compare_chunk_size: int = 512,
    device: str = "auto",
) -> tuple[list[int], list[dict[str, Any]]]:
    """Greedily keep earliest vectors and remove later cosine-near-duplicates.

    The input embeddings must already be L2-normalized. Matrix multiplication is
    performed in blocks to avoid materializing an N x N similarity matrix.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Semantic matching requires numpy and torch") from exc

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    count = matrix.shape[0]
    if count == 0:
        return [], []

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("embeddings contain zero-length vectors")
    if not np.allclose(norms, 1.0, atol=1e-3):
        matrix = matrix / norms[:, None]

    compute_device = resolve_device(device)
    tensor = torch.from_numpy(matrix).to(compute_device)
    retained_mask = [False] * count
    anchor_index = list(range(count))
    removed: list[dict[str, Any]] = []

    for start in range(0, count, compare_chunk_size):
        end = min(count, start + compare_chunk_size)
        block = tensor[start:end]

        if start > 0:
            similarities = block @ tensor[:start].T
            previous_scores, previous_indices = similarities.max(dim=1)
            previous_scores_cpu = previous_scores.detach().cpu().tolist()
            previous_indices_cpu = previous_indices.detach().cpu().tolist()
        else:
            previous_scores_cpu = [-1.0] * (end - start)
            previous_indices_cpu = [-1] * (end - start)

        for local_index, global_index in enumerate(range(start, end)):
            best_score = float(previous_scores_cpu[local_index])
            best_match = int(previous_indices_cpu[local_index])

            # The blockwise matrix above only sees earlier blocks. Compare with
            # earlier rows in the current block so duplicates are not missed.
            if local_index > 0:
                within_scores = block[local_index] @ block[:local_index].T
                within_score, within_index = within_scores.max(dim=0)
                within_score_value = float(within_score.detach().cpu().item())
                if within_score_value > best_score:
                    best_score = within_score_value
                    best_match = start + int(within_index.detach().cpu().item())

            if best_match >= 0 and best_score >= threshold:
                retained_anchor = anchor_index[best_match]
                anchor_index[global_index] = retained_anchor
                removed.append(
                    {
                        "duplicate_index": global_index,
                        "matched_index": best_match,
                        "retained_index": retained_anchor,
                        "similarity": round(best_score, 6),
                        "reason": "semantic_duplicate",
                    }
                )
            else:
                retained_mask[global_index] = True
                anchor_index[global_index] = global_index

        del block
        if compute_device.startswith("cuda"):
            torch.cuda.empty_cache()

    retained_indices = [index for index, keep in enumerate(retained_mask) if keep]
    return retained_indices, removed


def deduplicate_records(
    records: Sequence[dict[str, Any]],
    config: SemanticDedupConfig,
    *,
    precomputed_embeddings: "Any | None" = None,
    perform_exact_dedup: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run exact and semantic deduplication and return records/report/details."""
    config.validate()
    started = time.perf_counter()

    if perform_exact_dedup:
        exact_unique, exact_removed = exact_deduplicate(records)
    else:
        exact_unique = list(records)
        exact_removed = []

    texts: list[str] = []
    for index, record in enumerate(exact_unique):
        try:
            texts.append(extract_semantic_text(record, config.text_mode))
        except ValueError as exc:
            raise ValueError(f"Cannot extract semantic text from record {index}: {exc}") from exc

    embeddings = precomputed_embeddings
    if embeddings is None:
        embeddings = encode_with_bge(texts, config)
    elif len(embeddings) != len(exact_unique):
        raise ValueError(
            "precomputed_embeddings length does not match exact-unique records: "
            f"{len(embeddings)} != {len(exact_unique)}"
        )

    retained_indices, semantic_removed = cosine_greedy_deduplicate(
        embeddings,
        threshold=config.similarity_threshold,
        compare_chunk_size=config.compare_chunk_size,
        device=config.device,
    )
    retained = [exact_unique[index] for index in retained_indices]

    semantic_details: list[dict[str, Any]] = []
    for item in semantic_removed:
        detail = dict(item)
        duplicate_index = int(item["duplicate_index"])
        matched_index = int(item["matched_index"])
        retained_index = int(item["retained_index"])
        detail.update(
            {
                "duplicate_text": texts[duplicate_index],
                "matched_text": texts[matched_index],
                "retained_text": texts[retained_index],
                "record": exact_unique[duplicate_index],
            }
        )
        semantic_details.append(detail)

    total_removed = len(records) - len(retained)
    report = {
        "input_records": len(records),
        "exact_unique_records": len(exact_unique),
        "exact_duplicates_removed": len(exact_removed),
        "semantic_duplicates_removed": len(semantic_removed),
        "retained_records": len(retained),
        "total_removed": total_removed,
        "removal_rate": round(total_removed / len(records), 6) if records else 0.0,
        "semantic_removal_rate_after_exact": (
            round(len(semantic_removed) / len(exact_unique), 6) if exact_unique else 0.0
        ),
        "config": asdict(config),
        "resolved_device": resolve_device(config.device),
        "embedding_dimension": int(embeddings.shape[1]) if len(embeddings) else 0,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    all_removed = exact_removed + semantic_details
    return retained, report, all_removed


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dedup config JSON must be an object")
    return payload


def config_from_args(args: argparse.Namespace) -> SemanticDedupConfig:
    payload = load_config(args.config)
    defaults = asdict(SemanticDedupConfig())
    defaults.update(payload)
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            defaults[key] = value
    config = SemanticDedupConfig(**defaults)
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Normalized ShareGPT JSON/JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional JSON configuration")
    parser.add_argument("--model-name-or-path", dest="model_name_or_path")
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--text-mode", choices=VALID_TEXT_MODES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--compare-chunk-size", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    records = read_json_records(args.input)
    retained, report, removed = deduplicate_records(records, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "deduplicated.jsonl", retained)
    write_jsonl(args.output_dir / "removed_duplicates.jsonl", removed)
    (args.output_dir / "dedup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Retained data: {args.output_dir / 'deduplicated.jsonl'}")
    print(f"Removed details: {args.output_dir / 'removed_duplicates.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
