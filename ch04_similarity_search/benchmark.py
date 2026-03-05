from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from brute_force import brute_force_batch
from faiss_index import build_index, search as faiss_search
from metrics_bridge import recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    build_time_s: float
    query_total_s: float
    qps: float
    p50_latency_ms: float
    p95_latency_ms: float
    recall_at_k_mean: float
    mrr_at_k_mean: float
    ranked_doc_ids: list[list[str]]


def _latency_stats(latencies_s: list[float]) -> tuple[float, float]:
    if not latencies_s:
        return 0.0, 0.0
    latency_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    return float(np.percentile(latency_ms, 50)), float(np.percentile(latency_ms, 95))


def _quality(
    relevant_doc_ids: list[str],
    ranked_doc_ids: list[list[str]],
    k: int,
) -> tuple[float, float]:
    recalls: list[float] = []
    rrs: list[float] = []

    for relevant_doc_id, ranked in zip(relevant_doc_ids, ranked_doc_ids):
        topk = ranked[:k]
        recalls.append(recall_at_k(relevant_doc_id, topk))
        rrs.append(reciprocal_rank(relevant_doc_id, topk))

    return float(np.mean(recalls)), float(np.mean(rrs))


def benchmark_bruteforce(
    vectors: np.ndarray,
    doc_ids: list[str],
    queries: np.ndarray,
    relevant_doc_ids: list[str],
    k: int,
    mode: str,
) -> BenchmarkResult:
    build_time_s = 0.0

    latencies_s: list[float] = []
    ranked_doc_ids: list[list[str]] = []
    for query in queries:
        start = time.perf_counter()
        one = brute_force_batch(vectors=vectors, doc_ids=doc_ids, queries=np.asarray([query]), k=k, mode=mode)[0]
        latencies_s.append(time.perf_counter() - start)
        ranked_doc_ids.append(one)

    query_total_s = float(sum(latencies_s))
    p50_ms, p95_ms = _latency_stats(latencies_s)
    qps = 0.0 if query_total_s == 0 else len(queries) / query_total_s
    recall_mean, mrr_mean = _quality(relevant_doc_ids=relevant_doc_ids, ranked_doc_ids=ranked_doc_ids, k=k)

    return BenchmarkResult(
        name=f"Brute-force ({mode})",
        build_time_s=build_time_s,
        query_total_s=query_total_s,
        qps=qps,
        p50_latency_ms=p50_ms,
        p95_latency_ms=p95_ms,
        recall_at_k_mean=recall_mean,
        mrr_at_k_mean=mrr_mean,
        ranked_doc_ids=ranked_doc_ids,
    )


def benchmark_faiss(
    vectors: np.ndarray,
    doc_ids: list[str],
    queries: np.ndarray,
    relevant_doc_ids: list[str],
    k: int,
    index_type: str,
    nlist: int,
    nprobe: int,
) -> BenchmarkResult:
    build_start = time.perf_counter()
    index = build_index(vectors=vectors, index_type=index_type, nlist=nlist)
    build_time_s = time.perf_counter() - build_start

    # Warm up one small batch so benchmark reflects steady-state lookup more closely.
    warmup_count = min(5, len(queries))
    if warmup_count > 0:
        _ = faiss_search(index=index, doc_ids=doc_ids, queries=queries[:warmup_count], k=k, nprobe=nprobe)

    latencies_s: list[float] = []
    ranked_doc_ids: list[list[str]] = []
    for query in queries:
        start = time.perf_counter()
        one = faiss_search(
            index=index,
            doc_ids=doc_ids,
            queries=np.asarray([query], dtype=np.float32),
            k=k,
            nprobe=nprobe,
        )[0]
        latencies_s.append(time.perf_counter() - start)
        ranked_doc_ids.append(one)

    query_total_s = float(sum(latencies_s))
    p50_ms, p95_ms = _latency_stats(latencies_s)
    qps = 0.0 if query_total_s == 0 else len(queries) / query_total_s
    recall_mean, mrr_mean = _quality(relevant_doc_ids=relevant_doc_ids, ranked_doc_ids=ranked_doc_ids, k=k)

    return BenchmarkResult(
        name=f"FAISS {index_type.upper()}",
        build_time_s=build_time_s,
        query_total_s=query_total_s,
        qps=qps,
        p50_latency_ms=p50_ms,
        p95_latency_ms=p95_ms,
        recall_at_k_mean=recall_mean,
        mrr_at_k_mean=mrr_mean,
        ranked_doc_ids=ranked_doc_ids,
    )
