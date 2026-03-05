from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from ollama_provider import OllamaChatProvider

DEFAULT_MODEL = "llama3:8b-instruct-q4_0"


@dataclass
class LLMResult:
    answer: str
    usage: dict[str, Any]
    latency_s: float


def _flatten_messages(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", ""))
        lines.append(f"{role}:\n{content}")
    return "\n\n".join(lines)


def call_llm(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 256,
) -> LLMResult:
    _ = (temperature, max_tokens)
    selected_model = model or os.getenv("RAG_LLM_MODEL", DEFAULT_MODEL)
    provider = OllamaChatProvider()
    prompt = _flatten_messages(messages)

    started = time.perf_counter()
    try:
        result = provider.chat_with_metadata(prompt=prompt, model=selected_model)
    except Exception as exc:
        raise RuntimeError(
            "Failed to call local Ollama model. Ensure `ollama serve` is running and model "
            f"`{selected_model}` is pulled."
        ) from exc
    latency_s = time.perf_counter() - started

    return LLMResult(answer=result.get("text", ""), usage=result.get("usage", {}), latency_s=latency_s)
