from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float


def _words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) >= 3}


def _split_md_by_heading(text: str, source: str) -> list[RetrievedChunk]:
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())

    out: list[RetrievedChunk] = []
    for idx, chunk_text in enumerate(chunks, start=1):
        if chunk_text:
            out.append(RetrievedChunk(f"chunk_{idx}", source, chunk_text, 0.0))
    return out


def _split_fixed(text: str, source: str, chunk_chars: int = 700) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    i = 0
    cid = 1
    while i < len(text):
        part = text[i : i + chunk_chars].strip()
        if part:
            out.append(RetrievedChunk(f"chunk_{cid}", source, part, 0.0))
            cid += 1
        i += chunk_chars
    return out


def _load_chunks(data_dir: Path) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    policy_path = data_dir / "private_policy.md"
    rel_path = data_dir / "release_note_2026.txt"

    chunks.extend(_split_md_by_heading(policy_path.read_text(encoding="utf-8"), policy_path.name))
    chunks.extend(_split_fixed(rel_path.read_text(encoding="utf-8"), rel_path.name))

    # Reassign globally stable IDs after merge.
    for idx, chunk in enumerate(chunks, start=1):
        chunk.chunk_id = f"chunk_{idx}"

    return chunks


def retrieve_chunks(question: str, top_k: int = 3, data_dir: str | Path | None = None) -> list[RetrievedChunk]:
    base_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"
    chunks = _load_chunks(base_dir)
    q_terms = _words(question)

    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        overlap = q_terms.intersection(_words(chunk.text))
        score = len(overlap) / max(1, len(q_terms))
        scored.append(RetrievedChunk(chunk.chunk_id, chunk.source, chunk.text, score))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] ({c.source})\n{c.text}" for c in chunks)
