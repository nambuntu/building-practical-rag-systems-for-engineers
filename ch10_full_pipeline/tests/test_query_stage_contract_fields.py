from pathlib import Path
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from config import Settings  # noqa: E402
from domain.types import PreparedChunk, RetrievedChunk  # noqa: E402
from utils.io import write_json  # noqa: E402
from workflow.stages.query import run_query  # noqa: E402
from workflow.state import build_run_paths, ensure_run_dirs  # noqa: E402


class _FakeStore:
    def search(self, query_vector, top_k):
        _ = query_vector
        _ = top_k
        return [
            RetrievedChunk(
                chunk=PreparedChunk(chunk_id="c1", doc_id="d1", text="context text", metadata={}),
                score=0.7,
            )
        ]


class _FakeEmbedder:
    def __init__(self, model_path):
        _ = model_path

    def embed_query(self, text):
        _ = text
        return [0.1, 0.2]


class _FakeChat:
    def __init__(self, settings):
        _ = settings

    def chat(self, prompt):
        if "ALLOWED OUTPUTS" in prompt:
            return "ANSWER: grounded [1]"
        return "plain answer"


def test_query_writes_contract_fields(monkeypatch, tmp_path: Path):
    paths = build_run_paths(tmp_path, "run1")
    ensure_run_dirs(paths)
    write_json(paths.index_manifest_path, {"backend": "faiss", "embedding_model_path": "fake"})

    monkeypatch.setattr("workflow.stages.query.build_vector_store", lambda **kwargs: _FakeStore())
    monkeypatch.setattr(
        "workflow.stages.query.SentenceTransformerEmbeddingProvider",
        _FakeEmbedder,
    )
    monkeypatch.setattr("workflow.stages.query.OllamaChatProvider", _FakeChat)

    settings = Settings()
    result = run_query(
        paths,
        settings=settings,
        question="q",
        top_k=1,
        prompt_mode="contract",
        use_reranker=False,
        hybrid_retrieval=False,
        query_rewrite=False,
    )

    assert result.prompt_mode == "contract"
    assert result.citations == [1]
    assert result.refused is False
    assert result.format_ok is True


def test_query_prompt_mode_validation(monkeypatch, tmp_path: Path):
    paths = build_run_paths(tmp_path, "run2")
    ensure_run_dirs(paths)
    write_json(paths.index_manifest_path, {"backend": "faiss", "embedding_model_path": "fake"})

    settings = Settings()
    with pytest.raises(ValueError):
        run_query(
            paths,
            settings=settings,
            question="q",
            top_k=1,
            prompt_mode="bad",
            use_reranker=False,
            hybrid_retrieval=False,
            query_rewrite=False,
        )
