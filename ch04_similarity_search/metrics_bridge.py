def recall_at_k(relevant_doc_id: str, ranked_doc_ids: list[str]) -> float:
    return 1.0 if relevant_doc_id in ranked_doc_ids else 0.0


def reciprocal_rank(relevant_doc_id: str, ranked_doc_ids: list[str]) -> float:
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / index
    return 0.0
