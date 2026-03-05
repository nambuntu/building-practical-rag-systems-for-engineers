from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IngestRecord:
    doc_id: str
    title: str
    context: str
    question: str
    answers: list[str]
    source_split: str


@dataclass(slots=True)
class PreparedChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationExample:
    question: str
    gold_answers: list[str]
    doc_id: str


@dataclass(slots=True)
class RetrievedChunk:
    chunk: PreparedChunk
    score: float


@dataclass(slots=True)
class QueryResult:
    question: str
    rewritten_question: str
    answer: str
    retrieved: list[RetrievedChunk]
    prompt: str
    prompt_mode: str = "contract"
    citations: list[int] = field(default_factory=list)
    refused: bool = False
    format_ok: bool = False
    format_reason: str = ""


# Legacy types kept for compatibility with older modules.
@dataclass(slots=True)
class SentenceRecord:
    file_name: str
    sentence_number: int
    content: str
    embedding: list[float]


@dataclass(slots=True)
class SearchHit:
    id: int
    sentence_number: int
    content: str
    file_name: str
    distance: float


@dataclass(slots=True)
class ContextWindowRequest:
    file_name: str
    start_sentence_number: int
    end_sentence_number: int


@dataclass(slots=True)
class ContextSentence:
    id: int
    sentence_number: int
    content: str
    file_name: str


@dataclass(slots=True)
class ContextWindow:
    file_name: str
    start_sentence_number: int
    end_sentence_number: int
    sentences: list[ContextSentence] = field(default_factory=list)


@dataclass(slots=True)
class RagResult:
    query: str
    prompt: str
    answer: str
    contexts: list[ContextWindow]
