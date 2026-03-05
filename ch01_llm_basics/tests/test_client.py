import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import client  # noqa: E402


class DummyResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_client_success(monkeypatch, capsys):
    def fake_post(url, json, timeout):
        return DummyResponse(
            body={
                "model": json["model"],
                "response": "hello world",
                "prompt_eval_count": 11,
                "eval_count": 7,
            }
        )

    monkeypatch.setattr(client.requests, "post", fake_post)
    exit_code = client.main(["--prompt", "test"])
    assert exit_code == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["model"] == "llama3:latest"
    assert payload["response"] == "hello world"
    assert payload["seconds"] >= 0
    assert payload["prompt_eval_count"] == 11
    assert payload["eval_count"] == 7


def test_client_missing_optional_counts(monkeypatch, capsys):
    def fake_post(url, json, timeout):
        return DummyResponse(body={"response": "ok"})

    monkeypatch.setattr(client.requests, "post", fake_post)
    exit_code = client.main(["--prompt", "test"])
    assert exit_code == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["prompt_eval_count"] is None
    assert payload["eval_count"] is None


def test_client_connection_error(monkeypatch, capsys):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(client.requests, "post", fake_post)
    exit_code = client.main([])
    assert exit_code == 1

    err = capsys.readouterr().err
    assert "Could not reach Ollama" in err
