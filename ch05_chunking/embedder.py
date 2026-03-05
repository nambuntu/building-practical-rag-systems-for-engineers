from __future__ import annotations

import hashlib

import numpy as np

from chunking import tokenize


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % dim
    sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
    return bucket, sign


def embed_text(text: str, dim: int) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        bucket, sign = _bucket_and_sign(token, dim)
        vector[bucket] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector
