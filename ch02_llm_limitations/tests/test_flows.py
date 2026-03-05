import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ask_baseline  # noqa: E402
import ask_manual_inject  # noqa: E402
import ask_rag  # noqa: E402


def test_baseline_estimated_fallback(monkeypatch):
    class Fake:
        answer = "I do not know"
        usage = {}
        latency_s = 0.2

    monkeypatch.setattr(ask_baseline, "call_llm", lambda **kwargs: Fake())
    out = ask_baseline.run()
    assert out["estimated_tokens"] is True


def test_manual_overflow_skips_llm(monkeypatch):
    called = {"llm": False}

    class Fake:
        answer = "unused"
        usage = {}
        latency_s = 0.1

    def fake_call_llm(**kwargs):
        called["llm"] = True
        return Fake()

    monkeypatch.setattr(ask_manual_inject, "call_llm", fake_call_llm)
    out = ask_manual_inject.run(include_filler=True, prompt_limit=50)
    assert out["overflow"] is True
    assert called["llm"] is False


def test_rag_flow(monkeypatch):
    class Fake:
        answer = "Backup is paged after 7 minutes. Nimbus Mode reroutes reads. [chunk_1]"
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        latency_s = 0.3

    monkeypatch.setattr(ask_rag, "call_llm", lambda **kwargs: Fake())
    out = ask_rag.run(top_k=2, prompt_limit=5000)
    assert out["overflow"] is False
    assert len(out["retrieved_chunk_ids"]) == 2
