import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"
DEFAULT_TIMEOUT = 120.0
DEFAULT_OUTPUT = "results/results.csv"
CSV_FIELDS = ["model", "case", "seconds", "chars", "prompt_eval_count", "eval_count"]

PROMPT_CASES: list[tuple[str, str]] = [
    ("short_definition", "Define retrieval-augmented generation in one sentence."),
    ("numbered_list", "List exactly three uses of local LLMs."),
    ("tiny_summary", "Summarize: 'Small models can be practical' in five words."),
    ("format_check", "Answer with only: OK"),
    ("compare_brief", "Compare local and cloud LLMs in one short line."),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt benchmark against Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    return parser


def generate_non_stream(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    started = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{base_url}. Make sure Ollama is running (example: `ollama serve`)."
        ) from exc

    seconds = time.perf_counter() - started

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: {response.text.strip() or '<empty body>'}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("Malformed JSON response from Ollama.") from exc

    text = body.get("response", "")
    return {
        "model": body.get("model", model),
        "response": text,
        "seconds": round(seconds, 3),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    success_count = 0

    for case_name, prompt in PROMPT_CASES:
        started = time.perf_counter()
        try:
            result = generate_non_stream(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
            )
            success_count += 1
            rows.append(
                {
                    "model": result["model"],
                    "case": case_name,
                    "seconds": result["seconds"],
                    "chars": len(result["response"]),
                    "prompt_eval_count": result["prompt_eval_count"],
                    "eval_count": result["eval_count"],
                }
            )
        except RuntimeError as exc:
            elapsed = round(time.perf_counter() - started, 3)
            print(f"[warn] Case '{case_name}' failed: {exc}", file=sys.stderr)
            rows.append(
                {
                    "model": args.model,
                    "case": case_name,
                    "seconds": elapsed,
                    "chars": 0,
                    "prompt_eval_count": None,
                    "eval_count": None,
                }
            )

    output_path = Path(args.output)
    write_results(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")

    if success_count == 0:
        print("All benchmark cases failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
