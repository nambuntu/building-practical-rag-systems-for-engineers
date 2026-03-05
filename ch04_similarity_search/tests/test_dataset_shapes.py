import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset import build_dataset  # noqa: E402


def test_dataset_shapes_and_types_and_determinism():
    first = build_dataset(n=100, d=16, clusters=5, q=20, noise=0.02, seed=42)
    second = build_dataset(n=100, d=16, clusters=5, q=20, noise=0.02, seed=42)

    assert first.vectors.shape == (100, 16)
    assert first.queries.shape == (20, 16)
    assert first.vectors.dtype == np.float32
    assert first.queries.dtype == np.float32
    assert len(first.doc_ids) == 100
    assert len(first.relevant_doc_ids) == 20

    assert np.array_equal(first.vectors, second.vectors)
    assert np.array_equal(first.queries, second.queries)
    assert first.doc_ids == second.doc_ids
    assert first.relevant_doc_ids == second.relevant_doc_ids
