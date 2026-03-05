import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brute_force import brute_force_batch  # noqa: E402
from dataset import build_dataset  # noqa: E402
from faiss_index import build_index, search  # noqa: E402


def test_flat_matches_bruteforce_topk():
    dataset = build_dataset(n=120, d=32, clusters=6, q=12, noise=0.01, seed=7)
    k = 5

    brute_ranked = brute_force_batch(
        vectors=dataset.vectors,
        doc_ids=dataset.doc_ids,
        queries=dataset.queries,
        k=k,
        mode="vectorized",
    )

    index = build_index(vectors=dataset.vectors, index_type="flat", nlist=20)
    faiss_ranked = search(index=index, doc_ids=dataset.doc_ids, queries=dataset.queries, k=k, nprobe=8)

    assert faiss_ranked == brute_ranked
