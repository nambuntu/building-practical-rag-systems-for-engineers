from collections.abc import Iterable
from typing import Protocol

from domain.types import (
    ContextWindow,
    ContextWindowRequest,
    EvaluationExample,
    IngestRecord,
    PreparedChunk,
    RetrievedChunk,
    SearchHit,
    SentenceRecord,
)


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class ChatProvider(Protocol):
    def chat(self, prompt: str, *, model: str | None = None) -> str:
        ...


class CorpusSource(Protocol):
    def load_records(self, *, split: str, sample_size: int | None = None) -> Iterable[IngestRecord]:
        ...


class Evaluator(Protocol):
    def evaluate_retrieval(self, relevant_doc_id: str, ranked_doc_ids: list[str]) -> dict[str, float]:
        ...

    def evaluate_generation(self, prediction: str, gold_answers: list[str]) -> dict[str, float]:
        ...


class VectorStore(Protocol):
    def index(self, chunks: list[PreparedChunk], embeddings: list[list[float]]) -> int:
        ...

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        ...


# Legacy interface retained for compatibility.
class VectorRepository(Protocol):
    def insert_sentences(self, records: list[SentenceRecord]) -> int:
        ...

    def search(self, query_vector: list[float], limit: int) -> list[SearchHit]:
        ...

    def fetch_context_windows(self, requests: list[ContextWindowRequest]) -> list[ContextWindow]:
        ...


class EvaluationSource(Protocol):
    def load_examples(self, *, split: str, max_queries: int) -> list[EvaluationExample]:
        ...
