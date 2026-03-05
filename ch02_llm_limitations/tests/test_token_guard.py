import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import token_guard  # noqa: E402


def test_guard_ok():
    result = token_guard.check_prompt_budget([{"role": "user", "content": "hello"}], limit=999)
    assert result.ok is True


def test_guard_overflow_message_shape():
    result = token_guard.check_prompt_budget([{"role": "user", "content": "x" * 10000}], limit=10)
    assert result.ok is False
    assert result.message.startswith("CONTEXT_OVERFLOW: prompt=")
    assert " > limit=10" in result.message
