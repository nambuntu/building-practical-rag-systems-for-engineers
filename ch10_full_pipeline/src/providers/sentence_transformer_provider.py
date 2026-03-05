import os
from time import perf_counter

from sentence_transformers import SentenceTransformer

from domain.interfaces import EmbeddingProvider
from app_logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        started_at = perf_counter()
        self.model = SentenceTransformer(model_path, device=device)
        logger.info(
            "Embedding provider ready model=%s device=%s load_time=%.2fs",
            model_path,
            device,
            perf_counter() - started_at,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = perf_counter()
        batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "128"))
        batch_size = max(1, batch_size)
        logger.info("Embedding encode: batch_size=%s total_texts=%s", batch_size, len(texts))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_vectors = self.model.encode(texts[start:end], show_progress_bar=False)
            vectors.extend([vector.tolist() for vector in batch_vectors])
            if end % 512 == 0 or end == len(texts):
                logger.info(
                    "Embedding progress: %s/%s texts (%.1fs)",
                    end,
                    len(texts),
                    perf_counter() - started_at,
                )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
