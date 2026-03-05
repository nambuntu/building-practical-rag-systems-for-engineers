from pathlib import Path
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from config import Settings  # noqa: E402
from domain.types import QueryResult  # noqa: E402
from workflow.runner import run_pipeline  # noqa: E402


def test_runner_skip_eval_marks_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "workflow.runner.run_ingest",
        lambda *args, **kwargs: {"record_count": 1, "artifact": "a"},
    )
    monkeypatch.setattr(
        "workflow.runner.run_prepare",
        lambda *args, **kwargs: {"chunk_count": 1, "eval_examples": 1},
    )
    monkeypatch.setattr(
        "workflow.runner.run_index",
        lambda *args, **kwargs: {"indexed_chunks": 1, "backend": "faiss"},
    )
    monkeypatch.setattr(
        "workflow.runner.run_query",
        lambda *args, **kwargs: QueryResult(
            question="q",
            rewritten_question="q",
            answer="a",
            retrieved=[],
            prompt="p",
            prompt_mode="contract",
            citations=[1],
            refused=False,
            format_ok=True,
            format_reason="ok_answer",
        ),
    )

    called = {"evaluate": 0}

    def _eval(*args, **kwargs):
        called["evaluate"] += 1
        return {"recall_at_k": 1.0, "mrr": 1.0, "eval_mode": "retrieval"}

    monkeypatch.setattr("workflow.runner.run_evaluate", _eval)

    settings = Settings(runs_dir=str(tmp_path), vector_backend="faiss", profile="cpu_demo")
    _, manifest = run_pipeline(
        settings=settings,
        run_id="run1",
        profile_name="cpu_demo",
        question="q",
        source="local",
        split="train",
        sample_size=1,
        input_dir=str(tmp_path),
        backend="faiss",
        resume=False,
        eval_split="validation",
        eval_max_queries=10,
        with_eval=False,
        eval_mode="retrieval",
        prompt_mode="contract",
    )

    assert called["evaluate"] == 0
    assert manifest["stages"]["evaluate"]["status"] == "skipped"
