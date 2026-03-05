from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import Chunk
from run_lab import _metrics_for_k_values, _resolve_relevant_ids


def _chunks() -> list[Chunk]:
    return [
        Chunk("chunk_0", "coolant pump fails reset lever is marked j 17", 0, 8, 8),
        Chunk("chunk_1", "arm the ninety second timer", 9, 14, 5),
    ]


def test_resolve_relevant_ids_strict_and_loose() -> None:
    chunks = _chunks()
    strict = _resolve_relevant_ids(
        "strict", chunks, ["coolant pump fails", "reset lever is marked j 17"]
    )
    loose = _resolve_relevant_ids(
        "loose", chunks, ["coolant pump fails", "arm the ninety second timer"]
    )

    assert strict == {"chunk_0"}
    assert loose == {"chunk_0", "chunk_1"}


def test_resolve_relevant_ids_invalid_mode() -> None:
    with pytest.raises(ValueError, match="gold_mode"):
        _resolve_relevant_ids("bad", _chunks(), ["x"])


def test_metrics_for_k_values_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        _metrics_for_k_values(["chunk_0"], {"chunk_0"}, [1, 0, 3])


def test_metrics_for_k_values_unanswerable_path() -> None:
    out = _metrics_for_k_values(["chunk_0", "chunk_1"], set(), [1, 2])
    assert out[1] == (0.0, 0.0, 0.0)
    assert out[2] == (0.0, 0.0, 0.0)
