from __future__ import annotations

import ask_baseline
import ask_manual_inject
import ask_rag
from report import comparison_table, summarize_metrics


def main() -> int:
    print("=== Chapter 2 Lab: Why an LLM Alone Is Not Enough ===")

    print("\n[1] Baseline: no context")
    baseline = ask_baseline.run()
    print(f"Answer: {baseline['answer']}")
    print(summarize_metrics(baseline))
    print("Note: baseline may be vague because docs are private and dated 2026.")

    print("\n[2] Manual injection: private_policy + release_note")
    manual_small = ask_manual_inject.run(include_filler=False)
    if manual_small["overflow"]:
        print(manual_small["overflow_message"])
    else:
        print(f"Answer: {manual_small['answer']}")
        print(summarize_metrics(manual_small))

    print("\n[3] Manual injection: + filler_big (expected overflow)")
    manual_big = ask_manual_inject.run(include_filler=True)
    if manual_big["overflow"]:
        print(manual_big["overflow_message"])
        print("Tip: retrieve only relevant chunks")
    else:
        print(f"Answer: {manual_big['answer']}")
        print(summarize_metrics(manual_big))

    print("\n[4] Retrieval (lexical RAG)")
    rag = ask_rag.run()
    print(f"Retrieved: {', '.join(rag['retrieved_chunk_ids'])}")
    if rag["overflow"]:
        print(rag["overflow_message"])
    else:
        print(f"Answer: {rag['answer']}")
        print(summarize_metrics(rag))

    big_tokens = manual_big["total_tokens"]
    rag_tokens = rag["total_tokens"]
    token_delta = big_tokens - rag_tokens
    latency_delta = manual_big["latency_s"] - rag["latency_s"]
    print(
        "\nDelta (manual_inject_big vs rag): "
        f"{token_delta:+} tokens, {latency_delta:+.2f}s"
    )

    print("\nRetrieval vs fine-tuning")
    print(comparison_table())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
