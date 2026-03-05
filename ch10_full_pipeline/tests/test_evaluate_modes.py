from pathlib import Path
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from config import Settings  # noqa: E402
from domain.types import PreparedChunk, RetrievedChunk  # noqa: E402
from utils.io import write_json, write_jsonl  # noqa: E402
from workflow.stages.evaluate import run_evaluate  # noqa: E402
from workflow.state import build_run_paths, ensure_run_dirs  # noqa: E402


class _FakeStore:
    def search(self, query_vector, top_k):
        _ = query_vector
        _ = top_k
        return [RetrievedChunk(chunk=PreparedChunk("c1", "doc1", "ctx", {}), score=0.9)]


class _FakeEmbedder:
    def __init__(self, model_path):
        _ = model_path

    def embed_query(self, text):
        _ = text
        return [0.1, 0.2]


class _FakeChat:
    calls = 0

    def __init__(self, settings):
        _ = settings

    def chat(self, prompt):
        _ = prompt
        _FakeChat.calls += 1
        return "gold"


def _setup(paths):
    write_json(paths.index_manifest_path, {"backend": "faiss", "embedding_model_path": "fake"})
    write_jsonl(
        paths.prepare_eval_path,
        [{"question": "q", "gold_answers": ["gold"], "doc_id": "doc1"}],
    )


def test_retrieval_mode_skips_chat(monkeypatch, tmp_path: Path):
    paths = build_run_paths(tmp_path, "run1")
    ensure_run_dirs(paths)
    _setup(paths)

    monkeypatch.setattr("workflow.stages.evaluate.build_vector_store", lambda **kwargs: _FakeStore())
    monkeypatch.setattr("workflow.stages.evaluate.SentenceTransformerEmbeddingProvider", _FakeEmbedder)
    monkeypatch.setattr("workflow.stages.evaluate.OllamaChatProvider", _FakeChat)

    _FakeChat.calls = 0
    report = run_evaluate(
        paths,
        settings=Settings(),
        eval_split="validation",
        max_queries=1,
        top_k=1,
        eval_mode="retrieval",
    )
    assert report["eval_mode"] == "retrieval"
    assert "em" not in report
    assert _FakeChat.calls == 0


def test_full_mode_includes_em_f1(monkeypatch, tmp_path: Path):
    paths = build_run_paths(tmp_path, "run2")
    ensure_run_dirs(paths)
    _setup(paths)

    monkeypatch.setattr("workflow.stages.evaluate.build_vector_store", lambda **kwargs: _FakeStore())
    monkeypatch.setattr("workflow.stages.evaluate.SentenceTransformerEmbeddingProvider", _FakeEmbedder)
    monkeypatch.setattr("workflow.stages.evaluate.OllamaChatProvider", _FakeChat)

    report = run_evaluate(
        paths,
        settings=Settings(),
        eval_split="validation",
        max_queries=1,
        top_k=1,
        eval_mode="full",
    )
    assert report["eval_mode"] == "full"
    assert "em" in report and "f1" in report
