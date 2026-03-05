from pathlib import Path

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, asc, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from domain.interfaces import VectorStore
from domain.types import PreparedChunk, RetrievedChunk


class _Base(DeclarativeBase):
    pass


class _PgChunk(_Base):
    __tablename__ = "rag_chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, index=True)
    text: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector)


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str, index_dir: Path) -> None:
        self.engine = create_engine(database_url)
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        _Base.metadata.create_all(self.engine)

    def index(self, chunks: list[PreparedChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings sizes do not match.")

        with Session(self.engine) as session:
            session.query(_PgChunk).delete()
            for chunk, embedding in zip(chunks, embeddings):
                session.add(
                    _PgChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        metadata_json=__import__("json").dumps(chunk.metadata, ensure_ascii=False),
                        embedding=embedding,
                    )
                )
            session.commit()
        return len(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        with Session(self.engine) as session:
            distance = _PgChunk.embedding.cosine_distance(query_vector).label("distance")
            rows = session.execute(
                select(_PgChunk.chunk_id, _PgChunk.doc_id, _PgChunk.text, _PgChunk.metadata_json, distance)
                .order_by(asc(distance))
                .limit(top_k)
            ).all()

        return [
            RetrievedChunk(
                chunk=PreparedChunk(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    text=row.text,
                    metadata=__import__("json").loads(row.metadata_json),
                ),
                score=float(1.0 - row.distance),
            )
            for row in rows
        ]
