from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimension mismatch.")
    if not a:
        raise ValueError("Vectors must be non-empty.")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero-length vector norm is not allowed.")

    return dot / (norm_a * norm_b)


def build_matrix(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        raise ValueError("No vectors provided.")

    size = len(vectors)
    matrix: list[list[float]] = []
    for i in range(size):
        row: list[float] = []
        for j in range(size):
            row.append(cosine_similarity(vectors[i], vectors[j]))
        matrix.append(row)
    return matrix
