from __future__ import annotations

from dataclasses import dataclass
import re


WORD_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    token_start: int
    token_end: int
    token_count: int


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def fixed_token_chunks(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenize(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{index}",
                text=" ".join(chunk_tokens),
                token_start=start,
                token_end=end,
                token_count=len(chunk_tokens),
            )
        )
        index += 1
        start += step

    return chunks


def semantic_chunks(text: str, target_size: int) -> list[Chunk]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[Chunk] = []
    index = 0
    cursor = 0

    current_sentences: list[str] = []
    current_tokens = 0
    chunk_start = 0

    def flush() -> None:
        nonlocal index, cursor, current_sentences, current_tokens, chunk_start
        if not current_sentences:
            return

        chunk_text = " ".join(current_sentences).strip()
        token_count = len(tokenize(chunk_text))
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{index}",
                text=chunk_text,
                token_start=chunk_start,
                token_end=chunk_start + token_count,
                token_count=token_count,
            )
        )
        index += 1
        cursor = chunk_start + token_count
        current_sentences = []
        current_tokens = 0
        chunk_start = cursor

    for paragraph in paragraphs:
        for sentence in split_sentences(paragraph):
            sentence_tokens = tokenize(sentence)
            sentence_count = len(sentence_tokens)
            if sentence_count == 0:
                continue

            if not current_sentences:
                chunk_start = cursor

            if current_sentences and (current_tokens + sentence_count) > target_size:
                flush()
                chunk_start = cursor

            current_sentences.append(sentence)
            current_tokens += sentence_count

            # If a single sentence is already above target, keep it intact.
            if current_tokens >= target_size and len(current_sentences) == 1:
                flush()

    flush()
    return chunks
