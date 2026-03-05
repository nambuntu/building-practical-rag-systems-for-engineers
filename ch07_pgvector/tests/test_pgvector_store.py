from __future__ import annotations

from pathlib import Path
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import Chunk
from pgvector_store import PgVectorStore
from run_lab import _repeat_manual, _resolve_compare_faiss


class FakeCursor:
    def __init__(self, *, fetchall_rows=None, fetchone_row=None) -> None:
        self.fetchall_rows = fetchall_rows if fetchall_rows is not None else []
        self.fetchone_row = fetchone_row
        self.execute_calls: list[tuple[str, object]] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        self.execute_calls.append((sql, params))

    def executemany(self, sql: str, rows) -> None:
        self.executemany_calls.append((sql, list(rows)))

    def fetchall(self):
        return self.fetchall_rows

    def fetchone(self):
        return self.fetchone_row


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commit_calls = 0

    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_calls += 1


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"chunk_{idx}", text=f"text {idx}", token_start=0, token_end=1, token_count=1)
        for idx in range(n)
    ]


def test_constructor_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        PgVectorStore(dim=0)

    with pytest.raises(ValueError, match=r"vector\(1024\)"):
        PgVectorStore(dim=512)


def test_ensure_schema_executes_init_sql(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    init_sql = tmp_path / "00_init.sql"
    init_sql.write_text("SELECT 42;", encoding="utf-8")

    cursor = FakeCursor()
    conn = FakeConn(cursor)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: conn)

    PgVectorStore(dim=1024, init_sql_path=init_sql)

    assert any("SELECT 42;" in sql for sql, _ in cursor.execute_calls)
    assert conn.commit_calls == 1


def test_index_validates_and_inserts(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor()
    conn = FakeConn(cursor)
    monkeypatch.setattr(PgVectorStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: conn)

    store = PgVectorStore(dim=1024)

    with pytest.raises(ValueError, match="float32"):
        store.index(_chunks(1), np.zeros((1, 1024), dtype=np.float64))

    vectors = np.ones((2, 1024), dtype=np.float32)
    added = store.index(_chunks(2), vectors)

    assert added == 2
    assert len(cursor.executemany_calls) == 1
    sql, rows = cursor.executemany_calls[0]
    assert "ON CONFLICT (chunk_id)" in sql
    assert len(rows) == 2
    assert len(rows[0][2]) == 1024


def test_search_maps_rows_to_scored_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor(fetchall_rows=[("chunk_9", "hello world", 0.88)])
    conn = FakeConn(cursor)
    monkeypatch.setattr(PgVectorStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: conn)

    store = PgVectorStore(dim=1024)
    results = store.search(np.zeros(1024, dtype=np.float32), top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_9"
    assert results[0].text == "hello world"
    assert results[0].score == pytest.approx(0.88)


def test_search_rejects_bad_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PgVectorStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: FakeConn(FakeCursor()))
    store = PgVectorStore(dim=1024)

    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search(np.zeros(1024, dtype=np.float32), top_k=0)

    with pytest.raises(ValueError, match="query dim"):
        store.search(np.zeros(8, dtype=np.float32), top_k=1)


def test_create_ann_index_executes_expected_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor()
    conn = FakeConn(cursor)
    monkeypatch.setattr(PgVectorStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: conn)

    store = PgVectorStore(dim=1024)
    store.create_ann_index("hnsw")
    store.create_ann_index("ivfflat")

    statements = [sql.lower() for sql, _ in cursor.execute_calls]
    assert any("using hnsw" in sql for sql in statements)
    assert any("using ivfflat" in sql for sql in statements)
    assert any("analyze chunks" in sql for sql in statements)


def test_count_and_reset_use_expected_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = FakeCursor(fetchone_row=(7,))
    conn = FakeConn(cursor)
    monkeypatch.setattr(PgVectorStore, "ensure_schema", lambda self: None)
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: conn)

    store = PgVectorStore(dim=1024)
    assert store.count_chunks() == 7

    store.reset()
    statements = [sql.lower() for sql, _ in cursor.execute_calls]
    assert any("select count(*) from chunks" in sql for sql in statements)
    assert any("truncate table chunks" in sql for sql in statements)


def test_run_lab_helper_repeat_manual() -> None:
    assert _repeat_manual("abc", 1) == "abc"
    assert _repeat_manual("abc", 3) == "abc\n\nabc\n\nabc"


def test_run_lab_helper_resolve_compare() -> None:
    assert _resolve_compare_faiss(reuse_db=True, compare_faiss=True) == (False, True)
    assert _resolve_compare_faiss(reuse_db=True, compare_faiss=False) == (False, False)
    assert _resolve_compare_faiss(reuse_db=False, compare_faiss=True) == (True, False)
