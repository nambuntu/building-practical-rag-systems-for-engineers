from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prompts import build_contract_prompt, build_naive_prompt


def test_contract_prompt_contains_required_guardrails() -> None:
    prompt = build_contract_prompt(["ctx1", "ctx2"], "What happened?")
    assert "untrusted input" in prompt
    assert "ALLOWED OUTPUTS" in prompt
    assert "INSUFFICIENT_CONTEXT" in prompt
    assert "ANSWER:" in prompt
    assert "Context [1]" in prompt
    assert "Context [2]" in prompt


def test_naive_prompt_is_simple() -> None:
    prompt = build_naive_prompt(["ctx1"], "What happened?")
    assert "ALLOWED OUTPUTS" not in prompt
    assert "untrusted input" not in prompt
    assert "Context [1]" in prompt
