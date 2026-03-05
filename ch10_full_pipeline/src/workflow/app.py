from pathlib import Path
import os

import typer

from config import get_settings
from utils.io import read_json, read_jsonl
from workflow.profiles import PROFILES, get_profile
from workflow.runner import default_run_id, run_pipeline
from workflow.stages.evaluate import run_evaluate
from workflow.stages.index import run_index
from workflow.stages.ingest import run_ingest
from workflow.stages.prepare import run_prepare
from workflow.stages.query import run_query
from workflow.state import build_run_paths, ensure_run_dirs, load_manifest, save_manifest

app = typer.Typer(help="Stage-based RAG learning pipeline.")


@app.command("ingest")
def ingest_command(
    source: str = typer.Option("squad", help="Data source: squad | local"),
    split: str = typer.Option("train", help="Dataset split."),
    sample_size: int = typer.Option(5000, help="Limit for quick learning runs."),
    input_dir: str = typer.Option(None, help="Input directory when source=local."),
    run_id: str = typer.Option(None, help="Run ID; auto-generated if omitted."),
) -> None:
    settings = get_settings()
    resolved_run_id = run_id or default_run_id()
    paths = build_run_paths(Path(settings.runs_dir), resolved_run_id)
    ensure_run_dirs(paths)

    payload = run_ingest(
        paths,
        source=source,
        split=split,
        sample_size=sample_size,
        input_dir=input_dir,
    )
    manifest = load_manifest(paths)
    manifest.setdefault("stages", {})["ingest"] = {"status": "completed", "payload": payload}
    manifest["source"] = source
    save_manifest(paths, manifest)

    typer.echo(f"run_id={resolved_run_id}")
    typer.echo(f"ingested={payload['record_count']} artifact={payload['artifact']}")


@app.command("prepare")
def prepare_command(
    run_id: str = typer.Option(..., help="Run ID produced by ingest."),
    chunk_size: int = typer.Option(256, help="Chunk size in words."),
    chunk_overlap: int = typer.Option(40, help="Chunk overlap in words."),
) -> None:
    settings = get_settings()
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    ensure_run_dirs(paths)

    payload = run_prepare(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    manifest = load_manifest(paths)
    manifest.setdefault("stages", {})["prepare"] = {"status": "completed", "payload": payload}
    save_manifest(paths, manifest)

    typer.echo(f"chunks={payload['chunk_count']} eval_examples={payload['eval_examples']}")


@app.command("index")
def index_command(
    run_id: str = typer.Option(..., help="Run ID."),
    backend: str = typer.Option("faiss", help="Vector backend: faiss | pgvector"),
    embedding_model_path: str = typer.Option(None, help="Override embedding model path."),
) -> None:
    settings = get_settings()
    profile = get_profile(settings.profile if settings.profile in PROFILES else "cpu_demo")
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    ensure_run_dirs(paths)

    payload = run_index(
        paths,
        settings=settings,
        backend=backend,
        embedding_model_path=embedding_model_path or profile.embedding_model_path,
    )
    manifest = load_manifest(paths)
    manifest.setdefault("stages", {})["index"] = {"status": "completed", "payload": payload}
    manifest["backend"] = backend
    save_manifest(paths, manifest)

    typer.echo(f"indexed_chunks={payload['indexed_chunks']} backend={backend}")


@app.command("query")
def query_command(
    run_id: str = typer.Option(..., help="Run ID."),
    question: str = typer.Option(..., help="User question."),
    top_k: int = typer.Option(5, help="Top-k retrieval size."),
    prompt_mode: str = typer.Option("contract", help="Prompt mode: naive | contract"),
    use_reranker: bool = typer.Option(False, help="Enable reranking (placeholder)."),
    hybrid_retrieval: bool = typer.Option(False, help="Enable hybrid retrieval (placeholder)."),
    query_rewrite: bool = typer.Option(False, help="Enable query rewriting (placeholder)."),
) -> None:
    settings = get_settings()
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    ensure_run_dirs(paths)

    result = run_query(
        paths,
        settings=settings,
        question=question,
        top_k=top_k,
        prompt_mode=prompt_mode,
        use_reranker=use_reranker,
        hybrid_retrieval=hybrid_retrieval,
        query_rewrite=query_rewrite,
    )

    manifest = load_manifest(paths)
    manifest.setdefault("stages", {})["query"] = {
        "status": "completed",
        "payload": {
            "question": question,
            "prompt_mode": prompt_mode,
            "citations": result.citations,
            "refused": result.refused,
            "format_ok": result.format_ok,
            "result_file": str(paths.query_results_path),
        },
    }
    save_manifest(paths, manifest)

    typer.echo(result.answer)


@app.command("evaluate")
def evaluate_command(
    run_id: str = typer.Option(..., help="Run ID."),
    eval_split: str = typer.Option("validation", help="Evaluation split label for metadata."),
    max_queries: int = typer.Option(300, help="Maximum number of evaluation queries."),
    top_k: int = typer.Option(5, help="Top-k retrieval for metrics."),
    eval_mode: str = typer.Option("retrieval", help="Evaluation mode: retrieval | full"),
) -> None:
    settings = get_settings()
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    ensure_run_dirs(paths)

    report = run_evaluate(
        paths,
        settings=settings,
        eval_split=eval_split,
        max_queries=max_queries,
        top_k=top_k,
        eval_mode=eval_mode,
    )
    manifest = load_manifest(paths)
    manifest.setdefault("stages", {})["evaluate"] = {"status": "completed", "payload": report}
    save_manifest(paths, manifest)

    if report.get("eval_mode") == "full":
        typer.echo(
            f"Recall@k={report['recall_at_k']:.4f} "
            f"MRR={report['mrr']:.4f} "
            f"EM={report['em']:.4f} "
            f"F1={report['f1']:.4f}"
        )
    else:
        typer.echo(f"Recall@k={report['recall_at_k']:.4f} MRR={report['mrr']:.4f}")


@app.command("run")
def run_command(
    question: str = typer.Option(..., help="Question for end-to-end run."),
    profile: str = typer.Option("cpu_demo", help="Pipeline profile."),
    fast: bool = typer.Option(False, "--fast", help="Speed-focused mode for quick feedback."),
    source: str = typer.Option("squad", help="Source: squad | local"),
    split: str = typer.Option("train", help="Source split."),
    sample_size: int = typer.Option(5000, help="Sample size for ingest."),
    input_dir: str = typer.Option(None, help="Local input dir when source=local."),
    backend: str = typer.Option(None, help="Override backend."),
    run_id: str = typer.Option(None, help="Run id (auto if omitted)."),
    resume: bool = typer.Option(True, help="Resume completed stages when run exists."),
    eval_split: str = typer.Option("validation", help="Evaluation split label."),
    eval_max_queries: int = typer.Option(300, help="Evaluation query cap."),
    prompt_mode: str = typer.Option("contract", help="Prompt mode: naive | contract"),
    with_eval: bool = typer.Option(False, "--with-eval/--skip-eval", help="Run evaluate stage."),
    eval_mode: str = typer.Option("retrieval", help="Evaluation mode: retrieval | full"),
) -> None:
    settings = get_settings()
    resolved_run_id = run_id or default_run_id()
    selected_profile = profile
    selected_sample_size = sample_size
    selected_eval_max_queries = eval_max_queries

    if fast:
        if selected_profile == "cpu_demo":
            selected_profile = "cpu_quick"
        selected_sample_size = min(selected_sample_size, 1500)
        selected_eval_max_queries = min(selected_eval_max_queries, 30)
        typer.echo(
            "fast_mode=true "
            f"profile={selected_profile} "
            f"sample_size={selected_sample_size} "
            f"eval_max_queries={selected_eval_max_queries}"
        )

    if os.getenv("RAG_DATABASE_URL") and not backend and settings.vector_backend.lower() != "pgvector":
        typer.echo(
            "note=RAG_DATABASE_URL is set, but backend is still 'faiss'. "
            "Use --backend pgvector or set RAG_VECTOR_BACKEND=pgvector."
        )

    paths, manifest = run_pipeline(
        settings=settings,
        run_id=resolved_run_id,
        profile_name=selected_profile,
        question=question,
        source=source,
        split=split,
        sample_size=selected_sample_size,
        input_dir=input_dir,
        backend=backend,
        resume=resume,
        eval_split=eval_split,
        eval_max_queries=selected_eval_max_queries,
        with_eval=with_eval,
        eval_mode=eval_mode,
        prompt_mode=prompt_mode,
    )

    stage_payload = manifest.get("stages", {}).get("evaluate", {}).get("payload", {})
    query_payload = manifest.get("stages", {}).get("query", {}).get("payload", {})
    typer.echo(f"run_id={resolved_run_id}")
    typer.echo(f"manifest={paths.manifest_path}")
    typer.echo(f"backend={manifest.get('backend')}")
    if query_payload.get("answer"):
        typer.echo(f"answer={query_payload['answer']}")
    if "citations" in query_payload:
        typer.echo(f"citations={query_payload['citations']}")
    if "refused" in query_payload:
        typer.echo(f"refused={query_payload['refused']}")
    typer.echo(f"evaluate_status={manifest.get('stages', {}).get('evaluate', {}).get('status')}")
    if stage_payload:
        if stage_payload.get("eval_mode") == "full":
            typer.echo(
                f"Recall@k={stage_payload.get('recall_at_k', 0):.4f} "
                f"MRR={stage_payload.get('mrr', 0):.4f} "
                f"EM={stage_payload.get('em', 0):.4f} "
                f"F1={stage_payload.get('f1', 0):.4f}"
            )
        elif "recall_at_k" in stage_payload:
            typer.echo(
                f"Recall@k={stage_payload.get('recall_at_k', 0):.4f} "
                f"MRR={stage_payload.get('mrr', 0):.4f}"
            )


@app.command("show-manifest")
def show_manifest_command(run_id: str = typer.Option(..., help="Run ID.")) -> None:
    settings = get_settings()
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    manifest = read_json(paths.manifest_path)
    typer.echo(manifest)


@app.command("inspect")
def inspect_command(run_id: str = typer.Option(..., help="Run ID.")) -> None:
    settings = get_settings()
    paths = build_run_paths(Path(settings.runs_dir), run_id)
    manifest = read_json(paths.manifest_path)

    typer.echo(f"run_id={run_id}")
    typer.echo(f"manifest={paths.manifest_path}")
    typer.echo(f"raw={paths.ingest_raw_path}")
    typer.echo(f"chunks={paths.prepare_chunks_path}")
    typer.echo(f"index_manifest={paths.index_manifest_path}")
    typer.echo(f"query_results={paths.query_results_path}")
    typer.echo(f"evaluate_report={paths.eval_report_path}")
    typer.echo(f"run_log={paths.run_log_path}")

    if paths.query_results_path.exists():
        rows = read_jsonl(paths.query_results_path)
        if rows:
            latest = rows[-1]
            typer.echo(f"last_answer={latest.get('answer', '')}")
            typer.echo(f"last_citations={latest.get('citations', [])}")
            typer.echo(f"last_refused={latest.get('refused', False)}")

    evaluate_stage = manifest.get("stages", {}).get("evaluate", {})
    payload = evaluate_stage.get("payload", {})
    if payload:
        typer.echo(f"evaluate_status={evaluate_stage.get('status')}")
        typer.echo(f"eval_mode={payload.get('eval_mode', 'n/a')}")
        if "recall_at_k" in payload:
            typer.echo(f"recall_at_k={payload.get('recall_at_k')}")
        if "mrr" in payload:
            typer.echo(f"mrr={payload.get('mrr')}")
