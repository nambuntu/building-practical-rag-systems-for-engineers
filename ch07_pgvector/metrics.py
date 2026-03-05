from __future__ import annotations

from chunking import Chunk, tokenize


def recall_at_k(relevant_chunk_id: str, ranked_ids: list[str]) -> float:
    return 1.0 if relevant_chunk_id in ranked_ids else 0.0


def reciprocal_rank(relevant_chunk_id: str, ranked_ids: list[str]) -> float:
    for index, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id == relevant_chunk_id:
            return 1.0 / index
    return 0.0


def _normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def _contains_phrase(chunk_text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(chunk_text)


def find_relevant_chunk_id(chunks: list[Chunk], required_phrases: list[str]) -> str | None:
    for chunk in chunks:
        if all(_contains_phrase(chunk.text, phrase) for phrase in required_phrases):
            return chunk.chunk_id
    return None


def find_phrase_locations(chunks: list[Chunk], required_phrases: list[str]) -> dict[str, list[str]]:
    locations: dict[str, list[str]] = {}
    for phrase in required_phrases:
        locations[phrase] = [chunk.chunk_id for chunk in chunks if _contains_phrase(chunk.text, phrase)]
    return locations
