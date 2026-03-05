from __future__ import annotations

from dataclasses import dataclass

from tokens import estimate_message_tokens


@dataclass
class GuardResult:
    ok: bool
    prompt_tokens: int
    limit: int
    overflow_tokens: int
    message: str


def check_prompt_budget(messages: list[dict], limit: int) -> GuardResult:
    prompt_tokens = estimate_message_tokens(messages)
    overflow_tokens = max(0, prompt_tokens - limit)
    ok = overflow_tokens == 0
    if ok:
        return GuardResult(
            ok=True,
            prompt_tokens=prompt_tokens,
            limit=limit,
            overflow_tokens=0,
            message="OK",
        )

    return GuardResult(
        ok=False,
        prompt_tokens=prompt_tokens,
        limit=limit,
        overflow_tokens=overflow_tokens,
        message=f"CONTEXT_OVERFLOW: prompt={prompt_tokens} > limit={limit}",
    )
