from pathlib import Path
import pathlib
import sys

from typer.testing import CliRunner

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_settings  # noqa: E402
from utils.io import write_json, write_jsonl  # noqa: E402
from workflow.app import app  # noqa: E402
from workflow.state import build_run_paths, ensure_run_dirs  # noqa: E402


runner = CliRunner()


def test_query_prompt_mode_contract(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_RUNS_DIR", str(tmp_path))
    get_settings.cache_clear()

    class _Result:
        answer = "ok"
        citations = [1]
        refused = False
        format_ok = True

    monkeypatch.setattr("workflow.app.run_query", lambda *args, **kwargs: _Result())

    result = runner.invoke(
        app,
        [
            "query",
            "--run-id",
            "r1",
            "--question",
            "q",
            "--prompt-mode",
            "contract",
        ],
    )
    assert result.exit_code == 0


def test_evaluate_eval_mode_retrieval(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_RUNS_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "workflow.app.run_evaluate",
        lambda *args, **kwargs: {"recall_at_k": 0.5, "mrr": 0.4, "eval_mode": "retrieval"},
    )

    result = runner.invoke(
        app,
        ["evaluate", "--run-id", "r2", "--eval-mode", "retrieval"],
    )
    assert result.exit_code == 0
    assert "Recall@k" in result.stdout


def test_run_skip_eval(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_RUNS_DIR", str(tmp_path))
    get_settings.cache_clear()

    paths = build_run_paths(tmp_path, "r3")
    ensure_run_dirs(paths)
    write_json(paths.manifest_path, {"run_id": "r3", "backend": "faiss", "stages": {"evaluate": {"status": "skipped", "payload": {"reason": "with_eval=false"}}, "query": {"status": "completed", "payload": {"answer": "a"}}}})

    monkeypatch.setattr(
        "workflow.app.run_pipeline",
        lambda **kwargs: (paths, {"backend": "faiss", "stages": {"evaluate": {"status": "skipped", "payload": {"reason": "with_eval=false"}}, "query": {"status": "completed", "payload": {"answer": "a"}}}}),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--question",
            "q",
            "--source",
            "local",
            "--input-dir",
            str(tmp_path),
            "--skip-eval",
        ],
    )
    assert result.exit_code == 0
    assert "evaluate_status=skipped" in result.stdout


def test_inspect_command(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_RUNS_DIR", str(tmp_path))
    get_settings.cache_clear()

    paths = build_run_paths(tmp_path, "r4")
    ensure_run_dirs(paths)
    write_json(paths.manifest_path, {"run_id": "r4", "stages": {"evaluate": {"status": "completed", "payload": {"eval_mode": "retrieval", "recall_at_k": 1.0, "mrr": 1.0}}}})
    write_jsonl(paths.query_results_path, [{"answer": "A", "citations": [1], "refused": False}])

    result = runner.invoke(app, ["inspect", "--run-id", "r4"])
    assert result.exit_code == 0
    assert "last_citations" in result.stdout
