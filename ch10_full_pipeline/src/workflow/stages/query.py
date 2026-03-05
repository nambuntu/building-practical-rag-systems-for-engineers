from pathlib import Path
from time import perf_counter

from config import Settings
from domain.types import QueryResult, RetrievedChunk
from app_logging import get_logger
from services.prompting import (
    build_contract_prompt_from_chunks,
    build_naive_prompt_from_chunks,
    is_refusal,
    parse_citations,
    validate_contract_output,
)
from providers.ollama_provider import OllamaChatProvider
from providers.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from utils.io import read_json, write_jsonl
from vectorstores.factory import build_vector_store
from workflow.state import RunPaths

logger = get_logger(__name__)


def run_query(
    paths: RunPaths,
    *,
    settings: Settings,
    question: str,
    top_k: int,
    prompt_mode: str,
    use_reranker: bool,
    hybrid_retrieval: bool,
    query_rewrite: bool,
    write_result: bool = True,
) -> QueryResult:
    started_at = perf_counter()
    logger.info(
        "Query stage: top_k=%s prompt_mode=%s use_reranker=%s hybrid=%s rewrite=%s",
        top_k,
        prompt_mode,
        use_reranker,
        hybrid_retrieval,
        query_rewrite,
    )
    if prompt_mode not in {"naive", "contract"}:
        raise ValueError("prompt_mode must be 'naive' or 'contract'")

    index_manifest = read_json(paths.index_manifest_path)
    backend = str(index_manifest["backend"])

    retrieve_started = perf_counter()
    vector_store = build_vector_store(backend=backend, settings=settings, index_dir=Path(paths.root / "index"))
    embedding_provider = SentenceTransformerEmbeddingProvider(index_manifest["embedding_model_path"])

    rewritten_question = question.strip()
    if query_rewrite:
        rewritten_question = rewritten_question

    query_vector = embedding_provider.embed_query(rewritten_question)
    retrieved = vector_store.search(query_vector=query_vector, top_k=top_k)
    logger.info("Query stage: retrieved %s chunks in %.2fs", len(retrieved), perf_counter() - retrieve_started)

    if prompt_mode == "contract":
        prompt = build_contract_prompt_from_chunks(retrieved_chunks=retrieved, question=rewritten_question)
    else:
        prompt = build_naive_prompt_from_chunks(retrieved_chunks=retrieved, question=rewritten_question)

    chat_provider = OllamaChatProvider(settings)
    chat_started = perf_counter()
    answer = chat_provider.chat(prompt)
    logger.info("Query stage: generated answer in %.2fs", perf_counter() - chat_started)

    citations = parse_citations(answer)
    refused = is_refusal(answer)
    format_ok, format_reason = validate_contract_output(answer, max_context_id=len(retrieved))

    result = QueryResult(
        question=question,
        rewritten_question=rewritten_question,
        answer=answer,
        retrieved=retrieved,
        prompt=prompt,
        prompt_mode=prompt_mode,
        citations=citations,
        refused=refused,
        format_ok=format_ok,
        format_reason=format_reason,
    )

    if write_result:
        write_started = perf_counter()
        write_jsonl(
            paths.query_results_path,
            [
                {
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "prompt": prompt,
                    "top_k": top_k,
                    "prompt_mode": prompt_mode,
                    "use_reranker": use_reranker,
                    "hybrid_retrieval": hybrid_retrieval,
                    "query_rewrite": query_rewrite,
                    "citations": citations,
                    "refused": refused,
                    "format_ok": format_ok,
                    "format_reason": format_reason,
                    "retrieved": [
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "doc_id": item.chunk.doc_id,
                            "score": item.score,
                            "text": item.chunk.text,
                        }
                        for item in retrieved
                    ],
                }
            ],
        )
        logger.info("Query stage: wrote result file=%s in %.2fs", paths.query_results_path, perf_counter() - write_started)

    logger.info("Query stage: completed in %.2fs", perf_counter() - started_at)
    return result
