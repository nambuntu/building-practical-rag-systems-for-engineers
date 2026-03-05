from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from parse_and_check import extract_citations, is_refusal, validate_contract_output


def test_extract_citations_single_and_multi() -> None:
    text = "ANSWER: first [1] second [1,3]"
    assert extract_citations(text) == [1, 1, 3]


def test_extract_citations_repeated() -> None:
    text = "ANSWER: line one [2] line two [2]"
    assert extract_citations(text) == [2, 2]


def test_is_refusal_exact_token() -> None:
    assert is_refusal("INSUFFICIENT_CONTEXT") is True
    assert is_refusal(" INSUFFICIENT_CONTEXT  ") is True
    assert is_refusal("INSUFFICIENT_CONTEXT because...") is False


def test_validate_contract_valid_refusal() -> None:
    ok, reason = validate_contract_output("INSUFFICIENT_CONTEXT", max_context_id=3)
    assert ok is True
    assert reason == "ok_refusal"


def test_validate_contract_valid_answer() -> None:
    ok, reason = validate_contract_output("ANSWER: Do x. [1,2]", max_context_id=3)
    assert ok is True
    assert reason == "ok_answer"


def test_validate_contract_missing_prefix() -> None:
    ok, reason = validate_contract_output("Do x [1]", max_context_id=3)
    assert ok is False
    assert reason == "missing_answer_prefix"


def test_validate_contract_missing_citations() -> None:
    ok, reason = validate_contract_output("ANSWER: Do x", max_context_id=3)
    assert ok is False
    assert reason == "missing_citations"


def test_validate_contract_out_of_range_citation() -> None:
    ok, reason = validate_contract_output("ANSWER: Do x [4]", max_context_id=3)
    assert ok is False
    assert reason == "citation_out_of_range"
