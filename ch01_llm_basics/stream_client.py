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
DEFAULT_PROMPT = "Write one short sentence about local language models."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Ollama generate API (streaming).")
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


def stream_generate(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=True)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{base_url}. Make sure Ollama is running (example: `ollama serve`)."
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: {response.text.strip() or '<empty body>'}"
        )

    last_chunk: dict[str, Any] = {}
    wrote_text = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Malformed JSON chunk from Ollama stream.") from exc

        token = chunk.get("response", "")
        if token:
            print(token, end="", flush=True)
            wrote_text = True
        last_chunk = chunk

    if wrote_text:
        print()

    seconds = time.perf_counter() - started
    result = {
        "model": last_chunk.get("model", model),
        "seconds": round(seconds, 3),
        "prompt_eval_count": last_chunk.get("prompt_eval_count"),
        "eval_count": last_chunk.get("eval_count"),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = stream_generate(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Total duration: {result['seconds']:.3f}s")
    if result["prompt_eval_count"] is not None:
        print(f"prompt_eval_count: {result['prompt_eval_count']}")
    if result["eval_count"] is not None:
        print(f"eval_count: {result['eval_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
