import sqlite3
from pathlib import Path

import faiss
import numpy as np

from domain.interfaces import VectorStore
from domain.types import PreparedChunk, RetrievedChunk


class FaissSqliteVectorStore(VectorStore):
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.index_dir / "chunks.sqlite"
        self.faiss_path = self.index_dir / "index.faiss"

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                row_id INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        return conn

    def index(self, chunks: list[PreparedChunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings sizes do not match.")

        dim = len(embeddings[0])
        matrix = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, str(self.faiss_path))

        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks (row_id, chunk_id, doc_id, text, metadata_json) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        row_id,
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        __import__("json").dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for row_id, chunk in enumerate(chunks)
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return len(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.faiss_path}")

        index = faiss.read_index(str(self.faiss_path))
        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)

        scores, row_ids = index.search(query, top_k)
        flat_scores = scores[0].tolist()
        flat_row_ids = row_ids[0].tolist()

        conn = self._get_connection()
        try:
            results: list[RetrievedChunk] = []
            for row_id, score in zip(flat_row_ids, flat_scores):
                if row_id < 0:
                    continue
                row = conn.execute(
                    "SELECT chunk_id, doc_id, text, metadata_json FROM chunks WHERE row_id = ?",
                    (row_id,),
                ).fetchone()
                if row is None:
                    continue
                metadata = __import__("json").loads(row[3])
                chunk = PreparedChunk(chunk_id=row[0], doc_id=row[1], text=row[2], metadata=metadata)
                results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            return results
        finally:
            conn.close()
