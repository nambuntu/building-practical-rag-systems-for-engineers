import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import rag  # noqa: E402


def test_chunk_ids_stable(tmp_path):
    (tmp_path / "private_policy.md").write_text("# A\nalpha\n# B\nbeta", encoding="utf-8")
    (tmp_path / "release_note_2026.txt").write_text("nimbus mode text", encoding="utf-8")
    chunks = rag.retrieve_chunks("alpha", top_k=2, data_dir=tmp_path)
    assert chunks[0].chunk_id.startswith("chunk_")


def test_policy_question_retrieves_policy(tmp_path):
    (tmp_path / "private_policy.md").write_text("# Esc\nbackup paged at 7 minutes", encoding="utf-8")
    (tmp_path / "release_note_2026.txt").write_text("nimbus mode reroutes traffic", encoding="utf-8")
    top = rag.retrieve_chunks("when is backup paged", top_k=1, data_dir=tmp_path)[0]
    assert "backup" in top.text.lower()


def test_nimbus_question_retrieves_release_note(tmp_path):
    (tmp_path / "private_policy.md").write_text("# Policy\nsev details", encoding="utf-8")
    (tmp_path / "release_note_2026.txt").write_text("Nimbus Mode reroutes read traffic", encoding="utf-8")
    top = rag.retrieve_chunks("what did nimbus mode change", top_k=1, data_dir=tmp_path)[0]
    assert "nimbus" in top.text.lower()
