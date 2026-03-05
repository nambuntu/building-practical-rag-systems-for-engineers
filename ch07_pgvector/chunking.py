from __future__ import annotations

from dataclasses import dataclass
import re

WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    token_start: int
    token_end: int
    token_count: int


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


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
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{idx}",
                text=" ".join(chunk_tokens),
                token_start=start,
                token_end=end,
                token_count=len(chunk_tokens),
            )
        )
        idx += 1
        start += step

    return chunks
