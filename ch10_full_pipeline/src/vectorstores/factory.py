from pathlib import Path

from config import Settings
from domain.interfaces import VectorStore
from vectorstores.faiss_sqlite import FaissSqliteVectorStore
from vectorstores.pgvector_store import PgVectorStore


def build_vector_store(backend: str, *, settings: Settings, index_dir: Path) -> VectorStore:
    normalized = backend.lower()
    if normalized == "faiss":
        return FaissSqliteVectorStore(index_dir=index_dir)
    if normalized == "pgvector":
        return PgVectorStore(database_url=settings.database_url, index_dir=index_dir)
    raise ValueError(f"Unsupported backend '{backend}'. Expected one of: faiss, pgvector")
