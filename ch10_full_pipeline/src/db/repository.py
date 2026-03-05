from sqlalchemy import Select, and_, asc, select
from sqlalchemy.orm import Session

from db.models import TextEmbeddingModel
from domain.interfaces import VectorRepository
from domain.types import ContextSentence, ContextWindow, ContextWindowRequest, SearchHit, SentenceRecord


class SqlAlchemyVectorRepository(VectorRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_sentences(self, records: list[SentenceRecord]) -> int:
        models = [
            TextEmbeddingModel(
                file_name=record.file_name,
                sentence_number=record.sentence_number,
                content=record.content,
                embedding=record.embedding,
            )
            for record in records
        ]
        self.session.add_all(models)
        self.session.commit()
        return len(models)

    def _build_search_statement(self, query_vector: list[float], limit: int) -> Select:
        distance = TextEmbeddingModel.embedding.cosine_distance(query_vector).label("distance")
        return (
            select(
                TextEmbeddingModel.id,
                TextEmbeddingModel.sentence_number,
                TextEmbeddingModel.content,
                TextEmbeddingModel.file_name,
                distance,
            )
            .order_by(asc(distance))
            .limit(limit)
        )

    def search(self, query_vector: list[float], limit: int) -> list[SearchHit]:
        statement = self._build_search_statement(query_vector=query_vector, limit=limit)
        rows = self.session.execute(statement).all()
        return [
            SearchHit(
                id=row.id,
                sentence_number=row.sentence_number,
                content=row.content,
                file_name=row.file_name,
                distance=float(row.distance),
            )
            for row in rows
        ]

    def _build_window_statement(self, request: ContextWindowRequest) -> Select:
        return (
            select(
                TextEmbeddingModel.id,
                TextEmbeddingModel.sentence_number,
                TextEmbeddingModel.content,
                TextEmbeddingModel.file_name,
            )
            .where(
                and_(
                    TextEmbeddingModel.file_name == request.file_name,
                    TextEmbeddingModel.sentence_number >= request.start_sentence_number,
                    TextEmbeddingModel.sentence_number <= request.end_sentence_number,
                )
            )
            .order_by(asc(TextEmbeddingModel.sentence_number))
        )

    def fetch_context_windows(self, requests: list[ContextWindowRequest]) -> list[ContextWindow]:
        windows: list[ContextWindow] = []
        for request in requests:
            statement = self._build_window_statement(request)
            rows = self.session.execute(statement).all()
            sentences = [
                ContextSentence(
                    id=row.id,
                    sentence_number=row.sentence_number,
                    content=row.content,
                    file_name=row.file_name,
                )
                for row in rows
            ]
            windows.append(
                ContextWindow(
                    file_name=request.file_name,
                    start_sentence_number=request.start_sentence_number,
                    end_sentence_number=request.end_sentence_number,
                    sentences=sentences,
                )
            )
        return windows
