from __future__ import annotations

from typing import Any


def summarize_metrics(result: dict[str, Any]) -> str:
    est = " (estimated)" if result.get("estimated_tokens") else ""
    return (
        f"Tokens: in={result['prompt_tokens']} out={result['completion_tokens']} "
        f"total={result['total_tokens']}{est} | Latency={result['latency_s']:.2f}s"
    )


def comparison_table() -> str:
    lines = [
        "| Approach | Best for | Freshness | Private docs | Update speed | Typical failure |",
        "|---|---|---|---|---|---|",
        "| Retrieval (RAG) | ask with sources | high | yes | fast | bad retrieval / missing chunk |",
        "| Fine-tuning | style / task behavior | medium/low | risky | slow | overfit; stale facts |",
    ]
    return "\n".join(lines)
