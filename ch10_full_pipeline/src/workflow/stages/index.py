from pathlib import Path
from time import perf_counter

from config import Settings
from domain.types import PreparedChunk
from app_logging import get_logger
from providers.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from utils.io import read_jsonl, write_json
from vectorstores.factory import build_vector_store
from workflow.state import RunPaths

logger = get_logger(__name__)


def run_index(paths: RunPaths, *, settings: Settings, backend: str, embedding_model_path: str) -> dict[str, object]:
    started_at = perf_counter()
    logger.info("Index stage: backend=%s embedding_model=%s", backend, embedding_model_path)
    rows = read_jsonl(paths.prepare_chunks_path)
    chunks = [
        PreparedChunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            text=str(row["text"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]
    logger.info("Index stage: loaded %s chunks", len(chunks))

    embed_started = perf_counter()
    provider = SentenceTransformerEmbeddingProvider(embedding_model_path)
    embeddings = provider.embed_texts([chunk.text for chunk in chunks])
    logger.info("Index stage: generated %s embeddings in %.2fs", len(embeddings), perf_counter() - embed_started)

    index_dir = Path(paths.root / "index")
    index_started = perf_counter()
    vector_store = build_vector_store(backend=backend, settings=settings, index_dir=index_dir)
    count = vector_store.index(chunks=chunks, embeddings=embeddings)
    logger.info("Index stage: vector index wrote %s chunks in %.2fs", count, perf_counter() - index_started)

    payload = {
        "backend": backend,
        "embedding_model_path": embedding_model_path,
        "indexed_chunks": count,
    }
    write_json(paths.index_manifest_path, payload)
    logger.info("Index stage: completed in %.2fs manifest=%s", perf_counter() - started_at, paths.index_manifest_path)
    return payload
