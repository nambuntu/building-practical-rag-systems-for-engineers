from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    vectors: np.ndarray
    doc_ids: list[str]
    queries: np.ndarray
    relevant_doc_ids: list[str]


def build_dataset(n: int, d: int, clusters: int, q: int, noise: float, seed: int) -> Dataset:
    if n <= 0 or d <= 0 or clusters <= 0 or q <= 0:
        raise ValueError("n, d, clusters, and q must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    centroids = rng.normal(loc=0.0, scale=1.0, size=(clusters, d)).astype(np.float32)

    cluster_ids = rng.integers(0, clusters, size=n)
    vectors = centroids[cluster_ids] + rng.normal(loc=0.0, scale=0.1, size=(n, d)).astype(np.float32)

    doc_ids = [f"doc_{idx}" for idx in range(n)]

    base_indices = rng.integers(0, n, size=q)
    queries = vectors[base_indices] + rng.normal(loc=0.0, scale=noise, size=(q, d)).astype(np.float32)
    relevant_doc_ids = [doc_ids[idx] for idx in base_indices]

    return Dataset(vectors=vectors.astype(np.float32), doc_ids=doc_ids, queries=queries.astype(np.float32), relevant_doc_ids=relevant_doc_ids)


def save_dataset(path: str | Path, dataset: Dataset) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        vectors=dataset.vectors,
        doc_ids=np.asarray(dataset.doc_ids, dtype=object),
        queries=dataset.queries,
        relevant_doc_ids=np.asarray(dataset.relevant_doc_ids, dtype=object),
    )


def load_dataset(path: str | Path) -> Dataset:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file not found: {source}")

    with np.load(source, allow_pickle=True) as loaded:
        vectors = loaded["vectors"].astype(np.float32)
        queries = loaded["queries"].astype(np.float32)
        doc_ids = [str(item) for item in loaded["doc_ids"].tolist()]
        relevant_doc_ids = [str(item) for item in loaded["relevant_doc_ids"].tolist()]

    return Dataset(vectors=vectors, doc_ids=doc_ids, queries=queries, relevant_doc_ids=relevant_doc_ids)
