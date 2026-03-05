from collections.abc import Iterable
from time import perf_counter

from domain.types import EvaluationExample, PreparedChunk
from app_logging import get_logger
from utils.io import read_jsonl, write_jsonl
from workflow.state import RunPaths

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        yield " ".join(chunk_words)
        if end >= len(words):
            break


def run_prepare(paths: RunPaths, *, chunk_size: int, chunk_overlap: int) -> dict[str, object]:
    started_at = perf_counter()
    logger.info("Prepare stage: chunk_size=%s chunk_overlap=%s", chunk_size, chunk_overlap)
    raw_rows = read_jsonl(paths.ingest_raw_path)
    logger.info("Prepare stage: loaded %s raw rows", len(raw_rows))

    seen_contexts: dict[str, str] = {}
    chunked_doc_ids: set[str] = set()
    chunks: list[PreparedChunk] = []
    eval_examples: list[EvaluationExample] = []

    for idx, row in enumerate(raw_rows, start=1):
        context = str(row.get("context", "")).strip()
        if not context:
            continue

        doc_id = str(row["doc_id"])
        if context in seen_contexts:
            doc_id = seen_contexts[context]
        else:
            seen_contexts[context] = doc_id

        title = str(row.get("title", ""))
        source_split = str(row.get("source_split", ""))

        # Only chunk each canonical document context once.
        if doc_id not in chunked_doc_ids:
            for chunk_idx, chunk_text in enumerate(_chunk_text(context, chunk_size, chunk_overlap)):
                chunks.append(
                    PreparedChunk(
                        chunk_id=f"{doc_id}::chunk::{chunk_idx}",
                        doc_id=doc_id,
                        text=chunk_text,
                        metadata={"title": title, "source_split": source_split},
                    )
                )
            chunked_doc_ids.add(doc_id)

        question = str(row.get("question", "")).strip()
        answers = [str(answer) for answer in row.get("answers", []) if str(answer).strip()]
        if question and answers:
            eval_examples.append(EvaluationExample(question=question, gold_answers=answers, doc_id=doc_id))

        if idx % 1000 == 0:
            logger.info(
                "Prepare stage: processed %s/%s rows unique_docs=%s chunks=%s eval_examples=%s",
                idx,
                len(raw_rows),
                len(chunked_doc_ids),
                len(chunks),
                len(eval_examples),
            )

    write_started = perf_counter()
    write_jsonl(
        paths.prepare_chunks_path,
        [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    )
    write_jsonl(
        paths.prepare_eval_path,
        [
            {"question": item.question, "gold_answers": item.gold_answers, "doc_id": item.doc_id}
            for item in eval_examples
        ],
    )
    logger.info(
        "Prepare stage: wrote artifacts chunks=%s eval=%s in %.2fs (total %.2fs)",
        paths.prepare_chunks_path,
        paths.prepare_eval_path,
        perf_counter() - write_started,
        perf_counter() - started_at,
    )

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "unique_documents": len(set(chunk.doc_id for chunk in chunks)),
        "chunk_count": len(chunks),
        "eval_examples": len(eval_examples),
        "artifacts": [str(paths.prepare_chunks_path), str(paths.prepare_eval_path)],
    }
