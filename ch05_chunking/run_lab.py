from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from chunking import Chunk, fixed_token_chunks, semantic_chunks
from dataset import QueryCase, get_query_cases, load_manual_text
from embedder import embed_text
from metrics import find_phrase_locations, find_relevant_chunk_id, recall_at_k, reciprocal_rank
from report import RunResult, render_interpretation, render_query_debug, render_results_table


@dataclass(frozen=True)
class RunConfig:
    chunker: str
    chunk_size: int
    overlap: int

    @property
    def config_name(self) -> str:
        if self.chunker == "fixed":
            return f"fixed(size={self.chunk_size},overlap={self.overlap})"
        return f"semantic(size={self.chunk_size})"


def _build_chunks(text: str, config: RunConfig) -> list[Chunk]:
    if config.chunker == "fixed":
        return fixed_token_chunks(text=text, chunk_size=config.chunk_size, overlap=config.overlap)
    if config.chunker == "semantic":
        return semantic_chunks(text=text, target_size=config.chunk_size)
    raise ValueError("unknown chunker")


def _preset_grid() -> list[RunConfig]:
    return [
        RunConfig(chunker="fixed", chunk_size=80, overlap=0),
        RunConfig(chunker="fixed", chunk_size=80, overlap=40),
        RunConfig(chunker="fixed", chunk_size=180, overlap=0),
        RunConfig(chunker="fixed", chunk_size=180, overlap=40),
        RunConfig(chunker="fixed", chunk_size=360, overlap=0),
        RunConfig(chunker="semantic", chunk_size=180, overlap=0),
        RunConfig(chunker="semantic", chunk_size=360, overlap=0),
    ]


def _ranked_chunk_ids(matrix: np.ndarray, chunks: list[Chunk], query_text: str, dim: int, top_k: int) -> tuple[list[str], np.ndarray]:
    query_vector = embed_text(query_text, dim)
    scores = matrix @ query_vector
    order = np.argsort(-scores, kind="stable")[:top_k]
    ids = [chunks[idx].chunk_id for idx in order]
    return ids, scores[order]


def evaluate_config(
    text: str,
    queries: list[QueryCase],
    config: RunConfig,
    *,
    dim: int,
    top_k: int,
) -> tuple[RunResult, dict[str, dict[str, object]]]:
    chunks = _build_chunks(text=text, config=config)
    if not chunks:
        raise ValueError("Chunking produced zero chunks.")

    avg_tokens = float(np.mean([chunk.token_count for chunk in chunks]))

    embed_start = time.perf_counter()
    matrix = np.asarray([embed_text(chunk.text, dim) for chunk in chunks], dtype=np.float32)
    embed_s = time.perf_counter() - embed_start

    retrieve_start = time.perf_counter()
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    unanswerable_count = 0
    query_debug: dict[str, dict[str, object]] = {}

    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    for query in queries:
        gold_chunk_id = find_relevant_chunk_id(chunks, query.required_phrases)
        ranked_ids, top_scores = _ranked_chunk_ids(matrix, chunks, query.query_text, dim, top_k)

        if gold_chunk_id is None:
            unanswerable_count += 1
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
        else:
            recalls.append(recall_at_k(gold_chunk_id, ranked_ids))
            reciprocal_ranks.append(reciprocal_rank(gold_chunk_id, ranked_ids))

        ranked_items = []
        for rank, (chunk_id, score) in enumerate(zip(ranked_ids, top_scores.tolist()), start=1):
            ranked_items.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "text": by_id[chunk_id].text,
                }
            )

        query_debug[query.query_id] = {
            "gold_chunk_id": gold_chunk_id,
            "ranked_items": ranked_items,
            "phrase_locations": find_phrase_locations(chunks, query.required_phrases),
        }

    retrieve_s = time.perf_counter() - retrieve_start

    result = RunResult(
        config_name=config.config_name,
        chunker=config.chunker,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        chunk_count=len(chunks),
        avg_tokens=avg_tokens,
        embed_s=embed_s,
        retrieve_s=retrieve_s,
        recall_at_k=float(np.mean(recalls)),
        mrr_at_k=float(np.mean(reciprocal_ranks)),
        unanswerable_count=unanswerable_count,
    )
    return result, query_debug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chapter 5 chunking lab.")
    parser.add_argument("--chunker", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-query", type=str, default=None)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    _ = args.seed  # Reserved for compatibility; current lab path is deterministic.

    data_path = Path(__file__).resolve().parent / "data" / "sample_manual.txt"
    manual_text = load_manual_text(data_path)
    queries = get_query_cases()

    if args.grid:
        configs = _preset_grid()
    else:
        overlap = args.overlap if args.chunker == "fixed" else 0
        configs = [RunConfig(chunker=args.chunker, chunk_size=args.chunk_size, overlap=overlap)]

    all_results: list[RunResult] = []
    debug_store: dict[str, dict[str, dict[str, object]]] = {}

    for config in configs:
        result, query_debug = evaluate_config(
            text=manual_text,
            queries=queries,
            config=config,
            dim=args.dim,
            top_k=args.top_k,
        )
        all_results.append(result)
        debug_store[config.config_name] = query_debug

    print("=== Chapter 5 Lab: Chunking Tradeoffs ===")
    print(render_results_table(results=all_results, top_k=args.top_k))
    print(render_interpretation(all_results))

    if args.show_query:
        selected = next((query for query in queries if query.query_id == args.show_query), None)
        if selected is None:
            valid_ids = ", ".join(query.query_id for query in queries)
            print(f"Unknown query id: {args.show_query}. Valid IDs: {valid_ids}")
            return 1

        print("\n=== Query Debug ===")
        for result in all_results:
            info = debug_store[result.config_name][selected.query_id]
            print(
                render_query_debug(
                    query=selected,
                    config_name=result.config_name,
                    top_k=args.top_k,
                    gold_chunk_id=info["gold_chunk_id"],
                    phrase_locations=info["phrase_locations"],
                    ranked_items=info["ranked_items"],
                )
            )
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
