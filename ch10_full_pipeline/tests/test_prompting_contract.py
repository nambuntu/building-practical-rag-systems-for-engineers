import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from services.prompting import (  # noqa: E402
    build_contract_prompt_from_chunks,
    parse_citations,
    validate_contract_output,
)
from domain.types import PreparedChunk, RetrievedChunk  # noqa: E402


def _retrieved() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=PreparedChunk(chunk_id="c1", doc_id="d1", text="alpha", metadata={}),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=PreparedChunk(chunk_id="c2", doc_id="d1", text="beta", metadata={}),
            score=0.8,
        ),
    ]


def test_contract_prompt_has_guardrails():
    prompt = build_contract_prompt_from_chunks(_retrieved(), "What is alpha?")
    assert "untrusted input" in prompt
    assert "INSUFFICIENT_CONTEXT" in prompt
    assert "ANSWER:" in prompt


def test_parse_citations():
    assert parse_citations("ANSWER: x [1] y [1,3]") == [1, 1, 3]


def test_validate_contract_failures():
    ok, reason = validate_contract_output("text [1]", 3)
    assert ok is False
    assert reason == "missing_answer_prefix"

    ok, reason = validate_contract_output("ANSWER: text [4]", 3)
    assert ok is False
    assert reason == "citation_out_of_range"
