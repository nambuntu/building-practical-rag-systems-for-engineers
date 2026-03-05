import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brute_force import brute_force_topk  # noqa: E402


def test_identity_query_returns_self_for_both_modes():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    doc_ids = ["doc_0", "doc_1", "doc_2"]

    query = vectors[1]

    ranked_naive = brute_force_topk(vectors=vectors, doc_ids=doc_ids, query=query, k=2, mode="naive")
    ranked_vectorized = brute_force_topk(vectors=vectors, doc_ids=doc_ids, query=query, k=2, mode="vectorized")

    assert ranked_naive[0] == "doc_1"
    assert ranked_vectorized[0] == "doc_1"
