from __future__ import annotations

import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _rank_doc_ids(scores: np.ndarray, doc_ids: list[str], k: int) -> list[str]:
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[idx] for idx in order[:k]]


def _topk_naive(normalized_vectors: np.ndarray, doc_ids: list[str], query: np.ndarray, k: int) -> list[str]:
    scored: list[tuple[float, int]] = []
    for idx, vector in enumerate(normalized_vectors):
        scored.append((float(np.dot(vector, query)), idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc_ids[idx] for _, idx in scored[:k]]


def _topk_vectorized(normalized_vectors: np.ndarray, doc_ids: list[str], query: np.ndarray, k: int) -> list[str]:
    scores = normalized_vectors @ query
    return _rank_doc_ids(scores=scores, doc_ids=doc_ids, k=k)


def brute_force_topk(
    vectors: np.ndarray,
    doc_ids: list[str],
    query: np.ndarray,
    k: int,
    mode: str = "vectorized",
) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(doc_ids) != len(vectors):
        raise ValueError("doc_ids size must match vectors size.")

    normalized_vectors = _normalize_rows(vectors)
    normalized_query = _normalize_vector(query)

    if mode == "naive":
        return _topk_naive(normalized_vectors=normalized_vectors, doc_ids=doc_ids, query=normalized_query, k=k)
    if mode == "vectorized":
        return _topk_vectorized(normalized_vectors=normalized_vectors, doc_ids=doc_ids, query=normalized_query, k=k)

    raise ValueError("mode must be one of: naive, vectorized")


def brute_force_batch(
    vectors: np.ndarray,
    doc_ids: list[str],
    queries: np.ndarray,
    k: int,
    mode: str = "vectorized",
) -> list[list[str]]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(doc_ids) != len(vectors):
        raise ValueError("doc_ids size must match vectors size.")

    normalized_vectors = _normalize_rows(vectors)

    results: list[list[str]] = []
    for query in queries:
        normalized_query = _normalize_vector(query)
        if mode == "naive":
            ranked_ids = _topk_naive(normalized_vectors, doc_ids, normalized_query, k)
        elif mode == "vectorized":
            ranked_ids = _topk_vectorized(normalized_vectors, doc_ids, normalized_query, k)
        else:
            raise ValueError("mode must be one of: naive, vectorized")
        results.append(ranked_ids)
    return results
