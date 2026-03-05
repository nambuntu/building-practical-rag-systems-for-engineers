import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import stream_client  # noqa: E402


class DummyStreamResponse:
    def __init__(self, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def test_stream_success(monkeypatch, capsys):
    lines = [
        '{"response":"Hi ","done":false}',
        '{"response":"there","done":false}',
        '{"done":true,"prompt_eval_count":4,"eval_count":2}',
    ]

    def fake_post(url, json, timeout, stream):
        return DummyStreamResponse(lines=lines)

    monkeypatch.setattr(stream_client.requests, "post", fake_post)
    code = stream_client.main([])
    assert code == 0

    out = capsys.readouterr().out
    assert "Hi there" in out
    assert "Total duration:" in out
    assert "prompt_eval_count: 4" in out
    assert "eval_count: 2" in out


def test_stream_malformed_chunk(monkeypatch, capsys):
    lines = ['{"response":"ok"}', "not-json"]

    def fake_post(url, json, timeout, stream):
        return DummyStreamResponse(lines=lines)

    monkeypatch.setattr(stream_client.requests, "post", fake_post)
    code = stream_client.main([])
    assert code == 1

    err = capsys.readouterr().err
    assert "Malformed JSON chunk" in err
