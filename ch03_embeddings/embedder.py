from __future__ import annotations

from typing import Any

import requests


def get_embedding(text: str, model: str, base_url: str, timeout: float) -> list[float]:
    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it with `ollama serve` and confirm base URL."
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama embeddings API error {response.status_code}: {response.text.strip() or '<empty body>'}"
        )

    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned malformed JSON for embeddings.") from exc

    embedding = body.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(
            "Missing or empty embedding in response. Ensure model supports embeddings and is pulled locally."
        )

    if not all(isinstance(v, (int, float)) for v in embedding):
        raise RuntimeError("Embedding contains non-numeric values.")

    return [float(v) for v in embedding]
