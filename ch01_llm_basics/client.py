import argparse
import json
import os
import sys
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"
DEFAULT_TIMEOUT = 120.0
DEFAULT_PROMPT = "Return exactly five words about local LLMs."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Ollama generate API (non-streaming).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text.")
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
    return parser


def generate_once(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
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

    return {
        "model": body.get("model", model),
        "response": body.get("response", ""),
        "seconds": round(seconds, 3),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_once(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
