from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from chunking import Chunk
from vector_store import ScoredChunk


class PgVectorStore:
    def __init__(self, dim: int, init_sql_path: Path | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim != 1024:
            raise ValueError("this chapter schema uses vector(1024), so --dim must be 1024")

        self.dim = dim
        self.init_sql_path = init_sql_path or (Path(__file__).resolve().parent / "sql" / "00_init.sql")

        self._conn_kwargs = {
            "host": os.getenv("PGHOST", "localhost"),
            "port": int(os.getenv("PGPORT", "5432")),
            "dbname": os.getenv("PGDATABASE", "rag"),
            "user": os.getenv("PGUSER", "postgres"),
            "password": os.getenv("PGPASSWORD", "postgres"),
        }

        self.ensure_schema()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(**self._conn_kwargs)
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        sql_text = self.init_sql_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
            conn.commit()

    def count_chunks(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def reset(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
            conn.commit()

    def create_ann_index(self, kind: Literal["hnsw", "ivfflat"]) -> float:
        if kind not in {"hnsw", "ivfflat"}:
            raise ValueError("kind must be 'hnsw' or 'ivfflat'")

        sql = {
            "hnsw": (
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_ip_ops)"
            ),
            "ivfflat": (
                "CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat "
                "ON chunks USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)"
            ),
        }[kind]

        start = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if kind == "ivfflat":
                    cur.execute("ANALYZE chunks")
            conn.commit()
        return time.perf_counter() - start

    def index(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if vectors.dtype != np.float32:
            raise ValueError("vectors dtype must be float32")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vectors dim must be {self.dim}")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")

        rows = [
            (
                chunk.chunk_id,
                chunk.text,
                Vector(vector.astype(np.float32).tolist()),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (chunk_id, text, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                      text = EXCLUDED.text,
                      embedding = EXCLUDED.embedding
                    """,
                    rows,
                )
            conn.commit()

        return len(rows)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            if q.shape[0] != self.dim:
                raise ValueError(f"query dim must be {self.dim}")
            q = q.reshape(1, self.dim)
        elif q.ndim == 2:
            if q.shape != (1, self.dim):
                raise ValueError(f"query shape must be (1, {self.dim})")
        else:
            raise ValueError("query_vector must be 1D or 2D")

        vector = Vector(q[0].astype(np.float32).tolist())

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, text, -(embedding <#> %s) AS score
                    FROM chunks
                    ORDER BY embedding <#> %s
                    LIMIT %s
                    """,
                    (vector, vector, top_k),
                )
                rows = cur.fetchall()

        return [
            ScoredChunk(chunk_id=str(chunk_id), text=str(text), score=float(score))
            for chunk_id, text, score in rows
        ]
