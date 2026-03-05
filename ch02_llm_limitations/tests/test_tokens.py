import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tokens  # noqa: E402


def test_estimate_tokens_non_zero():
    assert tokens.estimate_tokens("hello") > 0


def test_estimate_messages_stable():
    messages = [{"role": "user", "content": "abc"}]
    assert tokens.estimate_message_tokens(messages) == tokens.estimate_message_tokens(messages)
