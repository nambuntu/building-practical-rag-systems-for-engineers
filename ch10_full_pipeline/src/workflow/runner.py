from datetime import datetime
from pathlib import Path
from time import perf_counter

from config import Settings
from app_logging import attach_file_logger
from app_logging import get_logger
from utils.io import read_json
from workflow.profiles import get_profile
from workflow.stages.evaluate import run_evaluate
from workflow.stages.index import run_index
from workflow.stages.ingest import run_ingest
from workflow.stages.prepare import run_prepare
from workflow.stages.query import run_query
from workflow.state import RunPaths, build_run_paths, ensure_run_dirs, load_manifest, save_manifest

logger = get_logger(__name__)


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _stage_complete(manifest: dict, stage: str) -> bool:
    return bool(manifest.get("stages", {}).get(stage, {}).get("status") == "completed")


def _mark_stage(manifest: dict, stage: str, payload: dict) -> None:
    manifest.setdefault("stages", {})[stage] = {"status": "completed", "payload": payload}


def run_pipeline(
    *,
    settings: Settings,
    run_id: str,
    profile_name: str,
    question: str,
    source: str,
    split: str,
    sample_size: int | None,
    input_dir: str | None,
    backend: str | None,
    resume: bool,
    eval_split: str,
    eval_max_queries: int,
    with_eval: bool,
    eval_mode: str,
    prompt_mode: str,
) -> tuple[RunPaths, dict]:
    started_at = perf_counter()
    profile = get_profile(profile_name)
    selected_backend = (backend or settings.vector_backend or profile.backend).lower()
    logger.info(
        "Pipeline start run_id=%s profile=%s source=%s split=%s sample_size=%s backend=%s resume=%s",
        run_id,
        profile.name,
        source,
        split,
        sample_size,
        selected_backend,
        resume,
    )

    paths = build_run_paths(Path(settings.runs_dir), run_id)
    ensure_run_dirs(paths)
    attach_file_logger(paths.run_log_path)
    manifest = load_manifest(paths)
    manifest["profile"] = profile.name
    manifest["source"] = source
    manifest["backend"] = selected_backend
    manifest["prompt_mode"] = prompt_mode
    manifest["with_eval"] = with_eval
    manifest["eval_mode"] = eval_mode

    if not resume or not _stage_complete(manifest, "ingest"):
        stage_started = perf_counter()
        logger.info("Stage ingest: started")
        ingest_payload = run_ingest(paths, source=source, split=split, sample_size=sample_size, input_dir=input_dir)
        _mark_stage(manifest, "ingest", ingest_payload)
        save_manifest(paths, manifest)
        logger.info(
            "Stage ingest: completed in %.2fs records=%s artifact=%s",
            perf_counter() - stage_started,
            ingest_payload["record_count"],
            ingest_payload["artifact"],
        )
    else:
        logger.info("Stage ingest: skipped (already completed, resume=true)")

    if not resume or not _stage_complete(manifest, "prepare"):
        stage_started = perf_counter()
        logger.info("Stage prepare: started")
        prepare_payload = run_prepare(paths, chunk_size=profile.chunk_size, chunk_overlap=profile.chunk_overlap)
        _mark_stage(manifest, "prepare", prepare_payload)
        save_manifest(paths, manifest)
        logger.info(
            "Stage prepare: completed in %.2fs chunks=%s eval_examples=%s",
            perf_counter() - stage_started,
            prepare_payload["chunk_count"],
            prepare_payload["eval_examples"],
        )
    else:
        logger.info("Stage prepare: skipped (already completed, resume=true)")

    if not resume or not _stage_complete(manifest, "index"):
        stage_started = perf_counter()
        logger.info("Stage index: started")
        index_payload = run_index(
            paths,
            settings=settings,
            backend=selected_backend,
            embedding_model_path=profile.embedding_model_path,
        )
        _mark_stage(manifest, "index", index_payload)
        save_manifest(paths, manifest)
        logger.info(
            "Stage index: completed in %.2fs indexed_chunks=%s backend=%s",
            perf_counter() - stage_started,
            index_payload["indexed_chunks"],
            index_payload["backend"],
        )
    else:
        logger.info("Stage index: skipped (already completed, resume=true)")

    stage_started = perf_counter()
    logger.info("Stage query: started question=%r top_k=%s", question, profile.top_k)
    query_result = run_query(
        paths,
        settings=settings,
        question=question,
        top_k=profile.top_k,
        prompt_mode=prompt_mode,
        use_reranker=False,
        hybrid_retrieval=False,
        query_rewrite=False,
    )
    _mark_stage(
        manifest,
        "query",
        {
            "question": query_result.question,
            "answer": query_result.answer,
            "prompt_mode": query_result.prompt_mode,
            "citations": query_result.citations,
            "refused": query_result.refused,
            "format_ok": query_result.format_ok,
            "result_file": str(paths.query_results_path),
        },
    )
    save_manifest(paths, manifest)
    logger.info("Stage query: completed in %.2fs", perf_counter() - stage_started)

    if with_eval:
        stage_started = perf_counter()
        logger.info(
            "Stage evaluate: started max_queries=%s top_k=%s eval_mode=%s",
            eval_max_queries,
            profile.top_k,
            eval_mode,
        )
        eval_payload = run_evaluate(
            paths,
            settings=settings,
            eval_split=eval_split,
            max_queries=eval_max_queries,
            top_k=profile.top_k,
            eval_mode=eval_mode,
        )
        _mark_stage(manifest, "evaluate", eval_payload)
        save_manifest(paths, manifest)
        logger.info(
            "Stage evaluate: completed in %.2fs recall@k=%.4f mrr=%.4f mode=%s",
            perf_counter() - stage_started,
            eval_payload["recall_at_k"],
            eval_payload["mrr"],
            eval_payload["eval_mode"],
        )
    else:
        manifest.setdefault("stages", {})["evaluate"] = {
            "status": "skipped",
            "payload": {"reason": "with_eval=false"},
        }
        save_manifest(paths, manifest)
        logger.info("Stage evaluate: skipped (with_eval=false)")

    logger.info("Pipeline completed in %.2fs", perf_counter() - started_at)

    return paths, read_json(paths.manifest_path)
