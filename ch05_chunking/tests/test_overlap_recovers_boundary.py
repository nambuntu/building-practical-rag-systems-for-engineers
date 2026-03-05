import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import fixed_token_chunks  # noqa: E402
from metrics import find_relevant_chunk_id, recall_at_k  # noqa: E402
from retrieval import retrieve_topk  # noqa: E402


def test_overlap_recovers_boundary_and_improves_recall():
    text = (
        "w0 w1 w2 w3 w4 coolant pump fails "
        "reset lever is marked j 17 w5 w6 w7"
    )
    required_phrases = ["coolant pump fails", "reset lever is marked j 17"]
    query = "coolant pump fails reset lever is marked j 17"

    chunks_no_overlap = fixed_token_chunks(text=text, chunk_size=8, overlap=0)
    gold_no_overlap = find_relevant_chunk_id(chunks_no_overlap, required_phrases)
    assert gold_no_overlap is None

    chunks_with_overlap = fixed_token_chunks(text=text, chunk_size=8, overlap=4)
    gold_with_overlap = find_relevant_chunk_id(chunks_with_overlap, required_phrases)
    assert gold_with_overlap is not None

    ranked = retrieve_topk(chunks=chunks_with_overlap, query=query, dim=128, top_k=3)
    ranked_ids = [item.chunk_id for item in ranked]
    assert recall_at_k(gold_with_overlap, ranked_ids) == 1.0
