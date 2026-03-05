from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from chunking import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float
    text: str


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if vectors.dtype != np.float32:
            raise ValueError("vectors dtype must be float32")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vectors dim must be {self.dim}")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")

        self._index.add(vectors)
        self._chunks.extend(chunks)
        return int(vectors.shape[0])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            if q.shape[0] != self.dim:
                raise ValueError(f"query dim must be {self.dim}")
            q = q.reshape(1, self.dim)
        elif q.ndim == 2:
            if q.shape != (1, self.dim):
                raise ValueError(f"query shape must be (1, {self.dim})")
        else:
            raise ValueError("query_vector must be 1D or 2D")

        scores, indices = self._index.search(q, top_k)
        results: list[ScoredChunk] = []

        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(ScoredChunk(chunk_id=chunk.chunk_id, score=float(score), text=chunk.text))

        return results
