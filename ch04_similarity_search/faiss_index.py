from __future__ import annotations

import faiss
import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(np.float32, copy=True)
    faiss.normalize_L2(out)
    return out


def build_index(vectors: np.ndarray, index_type: str, nlist: int = 200) -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")

    n, dim = vectors.shape
    matrix = _normalize_rows(vectors)

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index

    if index_type == "ivf":
        safe_nlist = max(1, min(int(nlist), int(n)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, safe_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        return index

    raise ValueError("index_type must be one of: flat, ivf")


def search(
    index: faiss.Index,
    doc_ids: list[str],
    queries: np.ndarray,
    k: int,
    nprobe: int = 8,
) -> list[list[str]]:
    if k <= 0:
        raise ValueError("k must be positive.")

    if hasattr(index, "nprobe"):
        index.nprobe = max(1, int(nprobe))

    query_matrix = _normalize_rows(queries)
    _, indices = index.search(query_matrix, k)

    results: list[list[str]] = []
    for row in indices:
        ranked = [doc_ids[idx] for idx in row.tolist() if idx >= 0]
        results.append(ranked)
    return results
