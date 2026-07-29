from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import semantic_deduplicate as MODULE  # noqa: E402


def sample(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
    }


def test_extract_first_user() -> None:
    record = sample("什么是高血压？", "血压持续偏高。")
    assert MODULE.extract_semantic_text(record, "first_user") == "什么是高血压？"


def test_cosine_greedy_keeps_earliest() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    retained, removed = MODULE.cosine_greedy_deduplicate(
        embeddings,
        threshold=0.95,
        compare_chunk_size=2,
        device="cpu",
    )
    assert retained == [0, 2]
    assert removed[0]["duplicate_index"] == 1
    assert removed[0]["retained_index"] == 0


def test_end_to_end_with_precomputed_embeddings() -> None:
    records = [
        sample("高血压是什么意思？", "回答一"),
        sample("请解释什么叫高血压", "回答二"),
        sample("糖尿病饮食注意什么？", "回答三"),
    ]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.995, 0.05], [0.0, 1.0]],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    config = MODULE.SemanticDedupConfig(
        similarity_threshold=0.95,
        device="cpu",
        compare_chunk_size=2,
    )
    retained, report, removed = MODULE.deduplicate_records(
        records,
        config,
        precomputed_embeddings=embeddings,
    )
    assert len(retained) == 2
    assert report["semantic_duplicates_removed"] == 1
    assert report["retained_records"] == 2
    assert removed[0]["duplicate_text"] == "请解释什么叫高血压"
