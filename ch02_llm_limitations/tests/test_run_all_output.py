import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_all  # noqa: E402


def test_run_all_sections(monkeypatch, capsys):
    monkeypatch.setattr(run_all.ask_baseline, "run", lambda: {
        "answer": "unknown",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "latency_s": 0.1,
        "estimated_tokens": False,
        "overflow": False,
        "retrieved_chunk_ids": [],
    })
    monkeypatch.setattr(run_all.ask_manual_inject, "run", lambda include_filler: {
        "answer": "ans" if not include_filler else "",
        "prompt_tokens": 100 if not include_filler else 4000,
        "completion_tokens": 20 if not include_filler else 0,
        "total_tokens": 120 if not include_filler else 4000,
        "latency_s": 1.0 if not include_filler else 0.0,
        "estimated_tokens": True,
        "overflow": include_filler,
        "overflow_message": "CONTEXT_OVERFLOW: prompt=4000 > limit=3000",
        "retrieved_chunk_ids": [],
    })
    monkeypatch.setattr(run_all.ask_rag, "run", lambda: {
        "answer": "rag ans",
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "latency_s": 0.5,
        "estimated_tokens": True,
        "overflow": False,
        "retrieved_chunk_ids": ["chunk_1"],
    })

    code = run_all.main()
    assert code == 0
    out = capsys.readouterr().out
    assert "[1] Baseline" in out
    assert "[4] Retrieval" in out
    assert "Retrieval vs fine-tuning" in out
