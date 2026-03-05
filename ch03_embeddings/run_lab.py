from __future__ import annotations

import argparse
import os
import sys

from embedder import get_embedding
from report import interpret, render_matrix
from similarity import build_matrix

DEFAULT_MODEL = "llama3:8b-instruct-q4_0"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0

SENTENCES: list[tuple[str, str]] = [
    ("db_fail", "The database connection failed."),
    ("db_connect", "The server could not connect to the database."),
    ("bananas", "Bananas are yellow and sweet."),
    ("crash_timeout", "The application crashed due to timeout."),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 3 embeddings demo lab.")
    parser.add_argument("--model", default=os.getenv("RAG_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    print("=== Chapter 3 Lab: Embeddings Without Math Pain ===")
    print(f"Model: {args.model}")

    labels: list[str] = []
    vectors: list[list[float]] = []

    for label, sentence in SENTENCES:
        try:
            vector = get_embedding(
                text=sentence,
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        labels.append(label)
        vectors.append(vector)
        preview = ", ".join(f"{v:.4f}" for v in vector[:5])
        print(f"- {label}: length={len(vector)} first5=[{preview}]")

    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        print("Embedding vector lengths are inconsistent.", file=sys.stderr)
        return 1

    try:
        matrix = build_matrix(vectors)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print()
    print(render_matrix(labels, matrix))
    print()
    print(interpret(labels, matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
