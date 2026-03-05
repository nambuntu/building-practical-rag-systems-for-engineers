from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import Chunk
from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunks_loose,
    relevant_chunks_strict,
)


def _chunks() -> list[Chunk]:
    return [
        Chunk("chunk_0", "coolant pump fails then reset lever is marked j 17", 0, 9, 9),
        Chunk("chunk_1", "arm the ninety second timer before restart", 10, 16, 6),
        Chunk("chunk_2", "quiet window starts at 23 00", 17, 23, 6),
    ]


def test_relevant_chunks_strict_requires_all_phrases() -> None:
    chunks = _chunks()
    ids = relevant_chunks_strict(chunks, ["coolant pump fails", "reset lever is marked j 17"])
    assert ids == ["chunk_0"]


def test_relevant_chunks_loose_accepts_any_phrase() -> None:
    chunks = _chunks()
    ids = relevant_chunks_loose(chunks, ["coolant pump fails", "arm the ninety second timer"])
    assert ids == ["chunk_0", "chunk_1"]


def test_recall_and_rr_hit_rank_1() -> None:
    relevant = {"chunk_2"}
    ranked = ["chunk_2", "chunk_0", "chunk_1"]
    assert recall_at_k(relevant, ranked) == 1.0
    assert reciprocal_rank(relevant, ranked) == 1.0


def test_recall_and_rr_hit_late() -> None:
    relevant = {"chunk_1"}
    ranked = ["chunk_2", "chunk_0", "chunk_1"]
    assert recall_at_k(relevant, ranked) == 1.0
    assert reciprocal_rank(relevant, ranked) == pytest.approx(1 / 3)


def test_recall_and_rr_miss_or_empty_relevant() -> None:
    assert recall_at_k({"chunk_x"}, ["chunk_0", "chunk_1"]) == 0.0
    assert reciprocal_rank({"chunk_x"}, ["chunk_0", "chunk_1"]) == 0.0
    assert recall_at_k(set(), ["chunk_0", "chunk_1"]) == 0.0
    assert reciprocal_rank(set(), ["chunk_0", "chunk_1"]) == 0.0


def test_precision_at_k_behavior() -> None:
    relevant = {"chunk_0", "chunk_2"}
    ranked = ["chunk_2", "chunk_1", "chunk_0"]
    assert precision_at_k(relevant, ranked, 1) == 1.0
    assert precision_at_k(relevant, ranked, 2) == 0.5
    assert precision_at_k(relevant, ranked, 3) == pytest.approx(2 / 3)


def test_precision_at_k_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k({"chunk_0"}, ["chunk_0"], 0)
