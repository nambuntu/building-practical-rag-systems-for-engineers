from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import psycopg
from rich.console import Console
from rich.table import Table

from chunking import Chunk, fixed_token_chunks, tokenize
from dataset import QueryCase, get_query_cases, load_manual_text
from embedder import embed_text
from metrics import find_relevant_chunk_id, recall_at_k, reciprocal_rank
from pgvector_store import PgVectorStore
from vector_store import FaissVectorStore


def _build_embeddings(chunks: list[Chunk], dim: int) -> np.ndarray:
    return np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)


def _preview(text: str, limit: int = 120) -> str:
    sample = text.replace("\n", " ").strip()
    if len(sample) <= limit:
        return sample
    return sample[: limit - 3] + "..."


def _print_hits(console: Console, label: str, results: list, top_display: int) -> None:
    console.print(f"  [bold]{label}[/bold]")
    for rank, result in enumerate(results[:top_display], start=1):
        console.print(f"    {rank}. {result.chunk_id} score={result.score:.4f} :: {_preview(result.text)}")


def _print_summary(
    console: Console,
    *,
    chunk_count: int,
    embed_s: float,
    pg_insert_s: float,
    pg_index_build_s: float,
    pg_search_total_s: float,
    faiss_index_build_s: float,
    faiss_search_total_s: float,
    recall_mean: float,
    mrr_mean: float,
    unanswerable_count: int,
    top_k: int,
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("chunk_count", str(chunk_count))
    table.add_row("embed_s", f"{embed_s:.6f}")
    table.add_row("pg_insert_s", f"{pg_insert_s:.6f}")
    table.add_row("pg_index_build_s", f"{pg_index_build_s:.6f}")
    table.add_row("pg_search_total_s", f"{pg_search_total_s:.6f}")
    table.add_row("faiss_index_build_s", f"{faiss_index_build_s:.6f}")
    table.add_row("faiss_search_total_s", f"{faiss_search_total_s:.6f}")
    table.add_row(f"Recall@{top_k}", f"{recall_mean:.4f}")
    table.add_row(f"MRR@{top_k}", f"{mrr_mean:.4f}")
    table.add_row("unanswerable_count", str(unanswerable_count))
    console.print("\n[bold]Summary[/bold]")
    console.print(table)


def _repeat_manual(text: str, repeat: int) -> str:
    if repeat <= 1:
        return text
    return "\n\n".join([text] * repeat)


def _resolve_compare_faiss(reuse_db: bool, compare_faiss: bool) -> tuple[bool, bool]:
    if reuse_db and compare_faiss:
        return False, True
    return bool(compare_faiss), False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 7: Persistent vector search with pgvector.")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pg-index", choices=["none", "hnsw", "ivfflat"], default="none")
    parser.add_argument("--compare-faiss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-db", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-db", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.dim != 1024:
        raise ValueError("Chapter 7 schema uses vector(1024). Please run with --dim 1024")

    console = Console()
    data_path = Path(__file__).resolve().parent / "data" / "sample_manual.txt"
    manual_text = _repeat_manual(load_manual_text(data_path), args.repeat)
    queries = get_query_cases()

    tokens = tokenize(manual_text)
    console.print("=== Chapter 7 Lab: Persistent Vector Store (pgvector + FAISS) ===")
    console.print(f"Loaded manual: repeat={args.repeat}, {len(tokens)} tokens")

    chunks = fixed_token_chunks(manual_text, chunk_size=args.chunk_size, overlap=args.overlap)
    console.print(f"Chunking: size={args.chunk_size} overlap={args.overlap} -> {len(chunks)} chunks")

    query_vectors = {query.query_id: embed_text(query.query_text, args.dim) for query in queries}

    try:
        pg_store = PgVectorStore(dim=args.dim)
    except psycopg.Error as exc:
        console.print("[red]Failed to connect to Postgres.[/red]")
        console.print("Hint: run `make up` in ch07_pgvector/ and verify PG* env vars.")
        console.print(f"Database error: {exc}")
        return 1

    if args.reset_db:
        pg_store.reset()
        console.print("Reset: truncated chunks table")

    compare_faiss, compare_warning = _resolve_compare_faiss(
        reuse_db=bool(args.reuse_db), compare_faiss=bool(args.compare_faiss)
    )
    embed_s = 0.0
    pg_insert_s = 0.0
    pg_index_build_s = 0.0
    faiss_index_build_s = 0.0
    faiss_search_total_s = 0.0
    matrix: np.ndarray | None = None

    if args.reuse_db:
        count = pg_store.count_chunks()
        if count == 0:
            console.print("[red]--reuse-db was set but chunks table is empty.[/red]")
            console.print("Run without --reuse-db (or run `make lab`) to build embeddings first.")
            return 1
        console.print(f"Reuse mode: skipping embedding + insert, using existing {count} DB rows")
        if compare_warning:
            console.print("FAISS compare disabled in --reuse-db mode because embeddings are skipped.")
    else:
        embed_start = time.perf_counter()
        matrix = _build_embeddings(chunks=chunks, dim=args.dim)
        embed_s = time.perf_counter() - embed_start
        console.print(f"Embedding: dim={args.dim} -> matrix shape {matrix.shape}")

        insert_start = time.perf_counter()
        added = pg_store.index(chunks=chunks, vectors=matrix)
        pg_insert_s = time.perf_counter() - insert_start
        console.print(f"pgvector insert/upsert: {added} rows in {pg_insert_s * 1000:.3f} ms")

        if args.pg_index != "none":
            pg_index_build_s = pg_store.create_ann_index(args.pg_index)
            console.print(
                f"pgvector ANN index ({args.pg_index}) built in {pg_index_build_s * 1000:.3f} ms"
            )

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    unanswerable_count = 0
    pg_search_total_s = 0.0

    if compare_faiss and matrix is not None:
        faiss_store = FaissVectorStore(dim=args.dim)
        faiss_start = time.perf_counter()
        faiss_store.index(chunks=chunks, vectors=matrix)
        faiss_index_build_s = time.perf_counter() - faiss_start
        console.print(f"FAISS index build: {faiss_index_build_s * 1000:.3f} ms")
    else:
        faiss_store = None

    for query in queries:
        qvec = query_vectors[query.query_id]

        pg_start = time.perf_counter()
        pg_results = pg_store.search(query_vector=qvec, top_k=args.top_k)
        pg_search_total_s += time.perf_counter() - pg_start

        pg_ranked_ids = [item.chunk_id for item in pg_results]
        gold_chunk_id = find_relevant_chunk_id(chunks, query.required_phrases)

        if gold_chunk_id is None:
            unanswerable_count += 1
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
        else:
            recalls.append(recall_at_k(gold_chunk_id, pg_ranked_ids))
            reciprocal_ranks.append(reciprocal_rank(gold_chunk_id, pg_ranked_ids))

        console.print(f"\n[bold]{query.query_id}[/bold]: {query.query_text}")
        console.print(f"  gold chunk: {gold_chunk_id if gold_chunk_id else 'NONE (unanswerable)'}")
        _print_hits(console, "pgvector", pg_results, top_display=min(3, len(pg_results)))

        if faiss_store is not None:
            faiss_start = time.perf_counter()
            faiss_results = faiss_store.search(query_vector=qvec, top_k=args.top_k)
            faiss_search_total_s += time.perf_counter() - faiss_start
            _print_hits(console, "FAISS", faiss_results, top_display=min(3, len(faiss_results)))

            faiss_ids = [item.chunk_id for item in faiss_results]
            overlap = len(set(pg_ranked_ids).intersection(faiss_ids))
            parity = "exact match" if pg_ranked_ids == faiss_ids else "different order/set"
            console.print(f"  overlap(pgvector,faiss): {overlap}/{args.top_k} ({parity})")

    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    mrr_mean = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    _print_summary(
        console,
        chunk_count=len(chunks),
        embed_s=embed_s,
        pg_insert_s=pg_insert_s,
        pg_index_build_s=pg_index_build_s,
        pg_search_total_s=pg_search_total_s,
        faiss_index_build_s=faiss_index_build_s,
        faiss_search_total_s=faiss_search_total_s,
        recall_mean=recall_mean,
        mrr_mean=mrr_mean,
        unanswerable_count=unanswerable_count,
        top_k=args.top_k,
    )

    console.print("\nInterpretation:")
    console.print("- pgvector keeps vectors durable across restarts and redeploys.")
    console.print("- FAISS is fast in-memory; pgvector adds network/SQL overhead and persistence benefits.")
    console.print("- ANN indexes trade index build time for faster search as datasets grow.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
