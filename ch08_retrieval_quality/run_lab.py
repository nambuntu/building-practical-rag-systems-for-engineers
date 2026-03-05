from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from chunking import Chunk, fixed_token_chunks, tokenize
from dataset import QueryCase, get_query_cases, load_manual_text
from embedder import embed_text
from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunks_loose,
    relevant_chunks_strict,
)
from vector_store import FaissVectorStore


def _build_embeddings(chunks: list[Chunk], dim: int) -> np.ndarray:
    return np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)


def _preview(text: str, limit: int = 120) -> str:
    sample = text.replace("\n", " ").strip()
    if len(sample) <= limit:
        return sample
    return sample[: limit - 3] + "..."


def _resolve_relevant_ids(gold_mode: str, chunks: list[Chunk], required_phrases: list[str]) -> set[str]:
    if gold_mode == "strict":
        return set(relevant_chunks_strict(chunks, required_phrases))
    if gold_mode == "loose":
        return set(relevant_chunks_loose(chunks, required_phrases))
    raise ValueError("gold_mode must be 'strict' or 'loose'")


def _metrics_for_k_values(
    ranked_ids: list[str], relevant_ids: set[str], ks: list[int]
) -> dict[int, tuple[float, float, float]]:
    values: dict[int, tuple[float, float, float]] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("all values in --ks must be positive")
        limited = ranked_ids[:k]
        values[k] = (
            recall_at_k(relevant_ids, limited),
            reciprocal_rank(relevant_ids, limited),
            precision_at_k(relevant_ids, limited, k),
        )
    return values


def _print_summary_tables(
    console: Console,
    *,
    chunk_count: int,
    embed_s: float,
    faiss_index_build_s: float,
    search_total_s: float,
    recall_mean: float,
    mrr_mean: float,
    precision_mean: float,
    unanswerable_count: int,
    top_k: int,
    by_k_metrics: dict[int, tuple[float, float, float]],
) -> None:
    summary = Table(show_header=True, header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("chunk_count", str(chunk_count))
    summary.add_row("embed_s", f"{embed_s:.6f}")
    summary.add_row("faiss_index_build_s", f"{faiss_index_build_s:.6f}")
    summary.add_row("search_total_s", f"{search_total_s:.6f}")
    summary.add_row(f"Recall@{top_k}", f"{recall_mean:.4f}")
    summary.add_row(f"MRR@{top_k}", f"{mrr_mean:.4f}")
    summary.add_row(f"naive_Precision@{top_k}", f"{precision_mean:.4f}")
    summary.add_row("unanswerable_count", str(unanswerable_count))

    by_k = Table(show_header=True, header_style="bold")
    by_k.add_column("k", justify="right")
    by_k.add_column("Recall@k", justify="right")
    by_k.add_column("MRR@k", justify="right")
    by_k.add_column("naive_Precision@k", justify="right")

    for k in sorted(by_k_metrics):
        recall_k, mrr_k, precision_k = by_k_metrics[k]
        by_k.add_row(str(k), f"{recall_k:.4f}", f"{mrr_k:.4f}", f"{precision_k:.4f}")

    console.print("\n[bold]Summary[/bold]")
    console.print(summary)
    console.print("\n[bold]Metrics by k[/bold]")
    console.print(by_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 8: Measure retrieval quality with FAISS.")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gold-mode", choices=["strict", "loose"], default="strict")
    parser.add_argument("--show-top", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.show_top <= 0:
        raise ValueError("--show-top must be positive")
    if any(k <= 0 for k in args.ks):
        raise ValueError("all values in --ks must be positive")

    random.seed(args.seed)

    console = Console()
    data_path = Path(__file__).resolve().parent / "data" / "sample_manual.txt"
    manual_text = load_manual_text(data_path)
    queries = get_query_cases()

    tokens = tokenize(manual_text)
    console.print("=== Chapter 8 Lab: Measuring Retrieval Quality ===")
    console.print(f"Loaded manual: 1 file, {len(tokens)} tokens")

    chunks = fixed_token_chunks(manual_text, chunk_size=args.chunk_size, overlap=args.overlap)
    console.print(f"Chunking: size={args.chunk_size} overlap={args.overlap} -> {len(chunks)} chunks")

    embed_start = time.perf_counter()
    matrix = _build_embeddings(chunks=chunks, dim=args.dim)
    embed_s = time.perf_counter() - embed_start
    console.print(f"Embedding: dim={args.dim} -> matrix shape {matrix.shape}")

    store = FaissVectorStore(dim=args.dim)
    index_start = time.perf_counter()
    store.index(chunks=chunks, vectors=matrix)
    faiss_index_build_s = time.perf_counter() - index_start
    console.print(f"FAISS index build: {faiss_index_build_s * 1000:.3f} ms")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    unanswerable_count = 0
    search_total_s = 0.0

    by_k_values: dict[int, dict[str, list[float]]] = {
        k: {"recall": [], "mrr": [], "precision": []} for k in sorted(set(args.ks))
    }

    for query in queries:
        query_vector = embed_text(query.query_text, args.dim)

        search_start = time.perf_counter()
        results = store.search(query_vector=query_vector, top_k=args.top_k)
        search_total_s += time.perf_counter() - search_start

        ranked_ids = [item.chunk_id for item in results]
        relevant_ids = _resolve_relevant_ids(args.gold_mode, chunks, query.required_phrases)

        if not relevant_ids:
            unanswerable_count += 1

        recall = recall_at_k(relevant_ids, ranked_ids)
        rr = reciprocal_rank(relevant_ids, ranked_ids)
        precision = precision_at_k(relevant_ids, ranked_ids, args.top_k)

        recalls.append(recall)
        reciprocal_ranks.append(rr)
        precisions.append(precision)

        metrics_by_k = _metrics_for_k_values(ranked_ids, relevant_ids, sorted(set(args.ks)))
        for k, (recall_k, mrr_k, precision_k) in metrics_by_k.items():
            by_k_values[k]["recall"].append(recall_k)
            by_k_values[k]["mrr"].append(mrr_k)
            by_k_values[k]["precision"].append(precision_k)

        console.print(f"\n[bold]{query.query_id}[/bold]: {query.query_text}")
        if relevant_ids:
            summary = ", ".join(sorted(relevant_ids))
            console.print(f"  relevant ({args.gold_mode}): {summary}")
        else:
            console.print(f"  relevant ({args.gold_mode}): NONE (unanswerable under current gold mode)")

        for rank, result in enumerate(results[: min(args.show_top, len(results))], start=1):
            console.print(
                f"  {rank}. {result.chunk_id} score={result.score:.4f} :: {_preview(result.text)}"
            )

        console.print(
            f"  metrics: Recall@{args.top_k}={recall:.4f} "
            f"RR@{args.top_k}={rr:.4f} naive_Precision@{args.top_k}={precision:.4f}"
        )

    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    mrr_mean = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    precision_mean = float(np.mean(precisions)) if precisions else 0.0

    by_k_metrics: dict[int, tuple[float, float, float]] = {}
    for k, values in by_k_values.items():
        by_k_metrics[k] = (
            float(np.mean(values["recall"])) if values["recall"] else 0.0,
            float(np.mean(values["mrr"])) if values["mrr"] else 0.0,
            float(np.mean(values["precision"])) if values["precision"] else 0.0,
        )

    _print_summary_tables(
        console,
        chunk_count=len(chunks),
        embed_s=embed_s,
        faiss_index_build_s=faiss_index_build_s,
        search_total_s=search_total_s,
        recall_mean=recall_mean,
        mrr_mean=mrr_mean,
        precision_mean=precision_mean,
        unanswerable_count=unanswerable_count,
        top_k=args.top_k,
        by_k_metrics=by_k_metrics,
    )

    console.print("\nInterpretation:")
    console.print("- Recall@k asks if retrieval found any relevant chunk.")
    console.print("- MRR@k rewards finding a relevant chunk earlier in ranking.")
    console.print("- naive_Precision@k is informative but not a primary retrieval optimization target.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
