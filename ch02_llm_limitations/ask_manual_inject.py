from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm import call_llm
from report import summarize_metrics
from token_guard import check_prompt_budget
from tokens import estimate_message_tokens, estimate_tokens

QUESTION = (
    "According to ACME On-Call Policy v2026-02-20, what is the escalation path for Sev-1 "
    "and when must the backup be paged? Also, in the 2026 release notes, what did feature "
    "'Nimbus Mode' change?"
)


def _usage_with_fallback(messages: list[dict], answer: str, usage: dict) -> tuple[dict, bool]:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    if prompt_tokens is not None and completion_tokens is not None:
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        total = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        }, False

    p = estimate_message_tokens(messages)
    c = estimate_tokens(answer)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}, True


def _read_context(include_filler: bool) -> str:
    data_dir = Path(__file__).resolve().parent / "data"
    files = ["private_policy.md", "release_note_2026.txt"]
    if include_filler:
        files.append("filler_big.txt")
    parts = []
    for file_name in files:
        text = (data_dir / file_name).read_text(encoding="utf-8")
        parts.append(f"## {file_name}\n{text}")
    return "\n\n".join(parts)


def run(
    question: str = QUESTION,
    model: str | None = None,
    include_filler: bool = False,
    prompt_limit: int | None = None,
) -> dict:
    limit = prompt_limit or int(os.getenv("CH2_PROMPT_TOKEN_LIMIT", "3000"))
    context = _read_context(include_filler=include_filler)
    messages = [
        {"role": "system", "content": "Use ONLY provided context. If missing, say you do not know."},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"},
    ]

    guard = check_prompt_budget(messages, limit=limit)
    if not guard.ok:
        return {
            "answer": "",
            "prompt_tokens": guard.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": guard.prompt_tokens,
            "latency_s": 0.0,
            "estimated_tokens": True,
            "overflow": True,
            "overflow_message": guard.message,
            "retrieved_chunk_ids": [],
        }

    result = call_llm(messages=messages, model=model, temperature=0, max_tokens=256)
    usage, estimated = _usage_with_fallback(messages, result.answer, result.usage)

    return {
        "answer": result.answer,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_s": result.latency_s,
        "estimated_tokens": estimated,
        "overflow": False,
        "retrieved_chunk_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 2 manual context injection run.")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--model", default=None)
    parser.add_argument("--include-filler", action="store_true")
    parser.add_argument("--prompt-limit", type=int, default=None)
    args = parser.parse_args(argv)

    output = run(
        question=args.question,
        model=args.model,
        include_filler=args.include_filler,
        prompt_limit=args.prompt_limit,
    )

    print("[Manual Injection]")
    if output["overflow"]:
        print(output["overflow_message"])
        print("Tip: retrieve only relevant chunks")
        return 0

    print(f"Answer: {output['answer']}")
    print(summarize_metrics(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
