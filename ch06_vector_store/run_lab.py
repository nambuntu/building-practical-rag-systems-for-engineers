from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from chunking import Chunk, fixed_token_chunks, tokenize
from dataset import QueryCase, get_query_cases, load_manual_text
from embedder import embed_text
from metrics import (
    find_phrase_locations,
    find_relevant_chunk_id,
    recall_at_k,
    reciprocal_rank,
)
from vector_store import FaissVectorStore


def _build_embeddings(chunks: list[Chunk], dim: int) -> np.ndarray:
    return np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)


def _bruteforce_topk(chunks: list[Chunk], matrix: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[str]:
    scores = matrix @ query_vector
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [chunks[idx].chunk_id for idx in order]


def _preview(text: str, limit: int = 120) -> str:
    sample = text.replace("\n", " ").strip()
    if len(sample) <= limit:
        return sample
    return sample[: limit - 3] + "..."


def _print_query_results(
    console: Console,
    query: QueryCase,
    results: list,
    gold_chunk_id: str | None,
    top_display: int,
) -> None:
    console.print(f"\n[bold]{query.query_id}[/bold]: {query.query_text}")
    console.print(f"  gold chunk: {gold_chunk_id if gold_chunk_id else 'NONE (unanswerable)'}")
    for rank, result in enumerate(results[:top_display], start=1):
        console.print(
            f"  {rank}. {result.chunk_id} score={result.score:.4f} :: {_preview(result.text)}"
        )


def _print_summary(
    console: Console,
    *,
    chunk_count: int,
    embed_s: float,
    index_build_s: float,
    query_total_s: float,
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
    table.add_row("index_build_s", f"{index_build_s:.6f}")
    table.add_row("search_total_s", f"{query_total_s:.6f}")
    table.add_row(f"Recall@{top_k}", f"{recall_mean:.4f}")
    table.add_row(f"MRR@{top_k}", f"{mrr_mean:.4f}")
    table.add_row("unanswerable_count", str(unanswerable_count))
    console.print("\n[bold]Summary[/bold]")
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 6: Build your first FAISS vector store.")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare-bruteforce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-query", type=str, default=None)
    args = parser.parse_args(argv)

    console = Console()
    data_path = Path(__file__).resolve().parent / "data" / "sample_manual.txt"
    manual_text = load_manual_text(data_path)
    queries = get_query_cases()

    tokens = tokenize(manual_text)
    console.print("=== Chapter 6 Lab: First Vector Store (FAISS) ===")
    console.print(f"Loaded manual: 1 file, {len(tokens)} tokens")

    chunks = fixed_token_chunks(manual_text, chunk_size=args.chunk_size, overlap=args.overlap)
    console.print(
        f"Chunking: size={args.chunk_size} overlap={args.overlap} -> {len(chunks)} chunks"
    )

    embed_start = time.perf_counter()
    matrix = _build_embeddings(chunks=chunks, dim=args.dim)
    embed_s = time.perf_counter() - embed_start
    console.print(f"Embedding: dim={args.dim} -> matrix shape {matrix.shape}")

    store = FaissVectorStore(dim=args.dim)
    index_start = time.perf_counter()
    added = store.index(chunks=chunks, vectors=matrix)
    index_build_s = time.perf_counter() - index_start
    console.print(f"Index built: {added} vectors in {index_build_s * 1000:.3f} ms")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    unanswerable_count = 0
    query_total_s = 0.0

    selected_debug_query: QueryCase | None = None

    for query in queries:
        query_vector = embed_text(query.query_text, args.dim)

        search_start = time.perf_counter()
        results = store.search(query_vector=query_vector, top_k=args.top_k)
        query_total_s += time.perf_counter() - search_start

        ranked_ids = [item.chunk_id for item in results]
        gold_chunk_id = find_relevant_chunk_id(chunks, query.required_phrases)

        if gold_chunk_id is None:
            unanswerable_count += 1
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
        else:
            recalls.append(recall_at_k(gold_chunk_id, ranked_ids))
            reciprocal_ranks.append(reciprocal_rank(gold_chunk_id, ranked_ids))

        _print_query_results(
            console=console,
            query=query,
            results=results,
            gold_chunk_id=gold_chunk_id,
            top_display=min(3, len(results)),
        )

        if args.compare_bruteforce:
            brute_ids = _bruteforce_topk(
                chunks=chunks,
                matrix=matrix,
                query_vector=query_vector,
                top_k=args.top_k,
            )
            overlap = len(set(ranked_ids).intersection(brute_ids))
            parity = "exact match" if ranked_ids == brute_ids else "different order/set"
            console.print(f"  brute-force overlap: {overlap}/{args.top_k} ({parity})")

        if args.show_query and query.query_id == args.show_query:
            selected_debug_query = query

    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    mrr_mean = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    _print_summary(
        console=console,
        chunk_count=len(chunks),
        embed_s=embed_s,
        index_build_s=index_build_s,
        query_total_s=query_total_s,
        recall_mean=recall_mean,
        mrr_mean=mrr_mean,
        unanswerable_count=unanswerable_count,
        top_k=args.top_k,
    )

    console.print("\nInterpretation:")
    if args.compare_bruteforce:
        console.print("- FAISS IndexFlatIP should match brute-force ranking on normalized vectors.")
    console.print("- Chunking parameters decide whether phrase-pair queries are answerable.")
    console.print("- With a small corpus, embedding time usually dominates search time.")

    if args.show_query:
        console.print("\n[bold]Show Query Debug[/bold]")
        if selected_debug_query is None:
            valid = ", ".join(item.query_id for item in queries)
            console.print(f"Unknown query id '{args.show_query}'. Valid IDs: {valid}")
            return 1

        gold_chunk_id = find_relevant_chunk_id(chunks, selected_debug_query.required_phrases)
        console.print(f"Query: {selected_debug_query.query_text}")
        console.print(f"Notes: {selected_debug_query.notes}")
        console.print(f"Required phrases: {', '.join(selected_debug_query.required_phrases)}")
        console.print(f"Gold chunk: {gold_chunk_id if gold_chunk_id else 'NONE (unanswerable)'}")

        if gold_chunk_id is None:
            locations = find_phrase_locations(chunks, selected_debug_query.required_phrases)
            console.print("Phrase locations:")
            for phrase, chunk_ids in locations.items():
                rendered = ", ".join(chunk_ids) if chunk_ids else "<not found>"
                console.print(f"- {phrase}: {rendered}")

        query_vector = embed_text(selected_debug_query.query_text, args.dim)
        results = store.search(query_vector=query_vector, top_k=args.top_k)
        console.print("Top hits:")
        for rank, item in enumerate(results, start=1):
            console.print(f"{rank}. {item.chunk_id} score={item.score:.4f} :: {_preview(item.text, 160)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
