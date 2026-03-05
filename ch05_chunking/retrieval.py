from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chunking import Chunk
from embedder import embed_text


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float
    text: str


def build_chunk_matrix(chunks: list[Chunk], dim: int) -> tuple[np.ndarray, list[str]]:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32), []

    matrix = np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)
    ids = [chunk.chunk_id for chunk in chunks]
    return matrix, ids


def retrieve_topk(chunks: list[Chunk], query: str, dim: int, top_k: int) -> list[ScoredChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not chunks:
        return []

    matrix, ids = build_chunk_matrix(chunks, dim)
    query_vector = embed_text(query, dim)
    scores = matrix @ query_vector
    order = np.argsort(-scores, kind="stable")[:top_k]

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        ScoredChunk(
            chunk_id=ids[idx],
            score=float(scores[idx]),
            text=by_id[ids[idx]].text,
        )
        for idx in order
    ]
