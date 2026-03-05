from pathlib import Path
from time import perf_counter

from config import Settings
from evaluation.metrics import exact_match, f1_score, recall_at_k, reciprocal_rank
from app_logging import get_logger
from utils.io import read_json, read_jsonl, write_json
from vectorstores.factory import build_vector_store
from providers.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from providers.ollama_provider import OllamaChatProvider
from workflow.state import RunPaths

logger = get_logger(__name__)


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts)
    return f"Context:\n{context_text}\n\nQuestion: {question}"


def run_evaluate(
    paths: RunPaths,
    *,
    settings: Settings,
    eval_split: str,
    max_queries: int,
    top_k: int,
    eval_mode: str,
) -> dict[str, object]:
    started_at = perf_counter()
    _ = eval_split
    eval_rows = read_jsonl(paths.prepare_eval_path)[:max_queries]
    index_manifest = read_json(paths.index_manifest_path)
    logger.info(
        "Evaluate stage: eval_split=%s requested_max_queries=%s actual_queries=%s top_k=%s eval_mode=%s",
        eval_split,
        max_queries,
        len(eval_rows),
        top_k,
        eval_mode,
    )
    if eval_mode not in {"retrieval", "full"}:
        raise ValueError("eval_mode must be 'retrieval' or 'full'")

    vector_store = build_vector_store(
        backend=str(index_manifest["backend"]),
        settings=settings,
        index_dir=Path(paths.root / "index"),
    )
    embedding_provider = SentenceTransformerEmbeddingProvider(index_manifest["embedding_model_path"])
    chat_provider = OllamaChatProvider(settings) if eval_mode == "full" else None

    retrieval_scores: list[float] = []
    reciprocal_scores: list[float] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []

    for idx, row in enumerate(eval_rows, start=1):
        question = str(row["question"])
        gold_answers = [str(item) for item in row.get("gold_answers", [])]
        relevant_doc_id = str(row["doc_id"])

        query_vector = embedding_provider.embed_query(question)
        retrieved = vector_store.search(query_vector=query_vector, top_k=top_k)
        ranked_doc_ids = [item.chunk.doc_id for item in retrieved]

        retrieval_scores.append(recall_at_k(relevant_doc_id, ranked_doc_ids))
        reciprocal_scores.append(reciprocal_rank(relevant_doc_id, ranked_doc_ids))

        if eval_mode == "full":
            assert chat_provider is not None
            contexts = [item.chunk.text for item in retrieved]
            prompt = _build_prompt(question, contexts)
            predicted = chat_provider.chat(prompt)
            em_scores.append(exact_match(predicted, gold_answers))
            f1_scores.append(f1_score(predicted, gold_answers))

        if idx % 10 == 0 or idx == len(eval_rows):
            elapsed = perf_counter() - started_at
            avg = elapsed / idx if idx else 0.0
            remaining = avg * (len(eval_rows) - idx)
            logger.info(
                "Evaluate stage: progress %s/%s elapsed=%.1fs eta=%.1fs",
                idx,
                len(eval_rows),
                elapsed,
                remaining,
            )

    count = len(eval_rows)
    report = {
        "evaluated_queries": count,
        "recall_at_k": (sum(retrieval_scores) / count) if count else 0.0,
        "mrr": (sum(reciprocal_scores) / count) if count else 0.0,
        "top_k": top_k,
        "eval_mode": eval_mode,
    }
    if eval_mode == "full":
        report["em"] = (sum(em_scores) / count) if count else 0.0
        report["f1"] = (sum(f1_scores) / count) if count else 0.0
    write_json(paths.eval_report_path, report)
    logger.info("Evaluate stage: completed in %.2fs report=%s", perf_counter() - started_at, paths.eval_report_path)
    return report
