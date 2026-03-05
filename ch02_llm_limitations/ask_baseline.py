from __future__ import annotations

import argparse

from llm import call_llm
from report import summarize_metrics
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


def run(question: str = QUESTION, model: str | None = None) -> dict:
    messages = [
        {"role": "system", "content": "If you do not have enough info, say you do not know."},
        {"role": "user", "content": question},
    ]
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
    parser = argparse.ArgumentParser(description="Chapter 2 baseline run (no context).")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    output = run(question=args.question, model=args.model)
    print("[Baseline]")
    print(f"Answer: {output['answer']}")
    print(summarize_metrics(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
