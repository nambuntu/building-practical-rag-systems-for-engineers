import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import Chunk  # noqa: E402
from embedder import embed_text  # noqa: E402
from vector_store import FaissVectorStore  # noqa: E402


def _sample_chunks() -> list[Chunk]:
    texts = [
        "coolant pump fails reset lever is marked j 17",
        "airflow drops below thirty cfm set fan speed to sixty percent",
        "quiet window starts at 23 00 do not purge the pressure lines",
        "say baton accepted on channel six write backup note on orange card",
        "run command open panel delta checksum should end with 44af",
    ]
    chunks: list[Chunk] = []
    for idx, text in enumerate(texts):
        tokens = text.split()
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{idx}",
                text=text,
                token_start=idx * 10,
                token_end=idx * 10 + len(tokens),
                token_count=len(tokens),
            )
        )
    return chunks


def test_search_parity_with_bruteforce():
    dim = 128
    chunks = _sample_chunks()
    matrix = np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)

    store = FaissVectorStore(dim=dim)
    store.index(chunks, matrix)

    query = "what lever resets the system when coolant pump fails"
    q = embed_text(query, dim)

    faiss_ids = [item.chunk_id for item in store.search(q, top_k=3)]

    scores = matrix @ q
    order = np.argsort(-scores, kind="stable")[:3]
    brute_ids = [chunks[idx].chunk_id for idx in order]

    assert faiss_ids == brute_ids


def test_input_validation():
    dim = 64
    chunks = _sample_chunks()[:2]
    matrix = np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)

    store = FaissVectorStore(dim=dim)

    try:
        store.index(chunks, matrix.astype(np.float64))
        assert False, "Expected dtype validation error"
    except ValueError:
        pass

    try:
        bad_dim = np.asarray([embed_text(chunk.text, dim + 1) for chunk in chunks], dtype=np.float32)
        store.index(chunks, bad_dim)
        assert False, "Expected dimension validation error"
    except ValueError:
        pass

    try:
        store.index(chunks[:1], matrix)
        assert False, "Expected length mismatch validation error"
    except ValueError:
        pass
