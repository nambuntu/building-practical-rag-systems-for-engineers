from __future__ import annotations

from chunking import Chunk, tokenize


def _normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def _contains_phrase(chunk_text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(chunk_text)


def relevant_chunks_strict(chunks: list[Chunk], required_phrases: list[str]) -> list[str]:
    if not required_phrases:
        return []
    return [
        chunk.chunk_id
        for chunk in chunks
        if all(_contains_phrase(chunk.text, phrase) for phrase in required_phrases)
    ]


def relevant_chunks_loose(chunks: list[Chunk], required_phrases: list[str]) -> list[str]:
    if not required_phrases:
        return []
    return [
        chunk.chunk_id
        for chunk in chunks
        if any(_contains_phrase(chunk.text, phrase) for phrase in required_phrases)
    ]


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(chunk_id in relevant_ids for chunk_id in ranked_ids) else 0.0


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked_ids:
        return 0.0

    limited = ranked_ids[:k]
    if not limited:
        return 0.0

    hits = sum(1 for chunk_id in limited if chunk_id in relevant_ids)
    return hits / len(limited)
