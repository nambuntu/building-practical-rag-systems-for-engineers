from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str) -> list[int]:
    citations: list[int] = []
    for match in CITATION_RE.finditer(text):
        parts = [item.strip() for item in match.group(1).split(",")]
        citations.extend(int(part) for part in parts if part)
    return citations


def is_refusal(text: str) -> bool:
    return text.strip() == "INSUFFICIENT_CONTEXT"


def validate_contract_output(text: str, max_context_id: int) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "empty_output"

    if stripped == "INSUFFICIENT_CONTEXT":
        return True, "ok_refusal"

    if not stripped.startswith("ANSWER:"):
        return False, "missing_answer_prefix"

    citations = extract_citations(stripped)
    if not citations:
        return False, "missing_citations"

    for citation in citations:
        if citation < 1 or citation > max_context_id:
            return False, "citation_out_of_range"

    return True, "ok_answer"
