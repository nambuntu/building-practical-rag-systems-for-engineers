import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import promptbench  # noqa: E402


def test_promptbench_writes_csv(monkeypatch, tmp_path, capsys):
    def fake_generate(base_url, model, prompt, timeout):
        return {
            "model": model,
            "response": "text",
            "seconds": 0.01,
            "prompt_eval_count": 1,
            "eval_count": 2,
        }

    monkeypatch.setattr(promptbench, "generate_non_stream", fake_generate)
    out_path = tmp_path / "results.csv"
    code = promptbench.main(["--output", str(out_path)])
    assert code == 0
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == promptbench.CSV_FIELDS
        assert len(rows) == len(promptbench.PROMPT_CASES)

    assert "Wrote" in capsys.readouterr().out


def test_promptbench_partial_failure(monkeypatch, tmp_path):
    state = {"count": 0}

    def fake_generate(base_url, model, prompt, timeout):
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("temporary")
        return {
            "model": model,
            "response": "ok",
            "seconds": 0.01,
            "prompt_eval_count": None,
            "eval_count": None,
        }

    monkeypatch.setattr(promptbench, "generate_non_stream", fake_generate)
    out_path = tmp_path / "partial.csv"
    code = promptbench.main(["--output", str(out_path)])
    assert code == 0

    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(promptbench.PROMPT_CASES)
    assert rows[0]["chars"] == "0"


def test_promptbench_all_failures(monkeypatch, tmp_path, capsys):
    def fake_generate(base_url, model, prompt, timeout):
        raise RuntimeError("down")

    monkeypatch.setattr(promptbench, "generate_non_stream", fake_generate)
    out_path = tmp_path / "all_fail.csv"
    code = promptbench.main(["--output", str(out_path)])
    assert code == 1

    err = capsys.readouterr().err
    assert "All benchmark cases failed." in err
