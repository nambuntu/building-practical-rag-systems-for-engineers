from utils.io import write_jsonl, read_jsonl
from workflow.stages.prepare import run_prepare
from workflow.state import build_run_paths, ensure_run_dirs


def test_prepare_deduplicates_same_context(tmp_path):
    paths = build_run_paths(tmp_path, "run1")
    ensure_run_dirs(paths)

    write_jsonl(
        paths.ingest_raw_path,
        [
            {
                "doc_id": "a",
                "title": "A",
                "context": "same context text for both",
                "question": "q1",
                "answers": ["ans"],
                "source_split": "train",
            },
            {
                "doc_id": "b",
                "title": "B",
                "context": "same context text for both",
                "question": "q2",
                "answers": ["ans2"],
                "source_split": "train",
            },
        ],
    )

    payload = run_prepare(paths, chunk_size=10, chunk_overlap=2)

    chunks = read_jsonl(paths.prepare_chunks_path)
    doc_ids = {item["doc_id"] for item in chunks}
    chunk_ids = [item["chunk_id"] for item in chunks]

    assert payload["chunk_count"] >= 1
    assert doc_ids == {"a"}
    assert len(chunk_ids) == len(set(chunk_ids))
