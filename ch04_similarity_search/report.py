from __future__ import annotations

from typing import Any

from benchmark import BenchmarkResult


def _line_for_result(result: BenchmarkResult, k: int) -> list[str]:
    return [
        f"{result.name}:",
        f"  build: {result.build_time_s:.4f}s",
        f"  query total: {result.query_total_s:.4f}s",
        f"  qps: {result.qps:.2f}",
        f"  latency p50/p95: {result.p50_latency_ms:.3f}/{result.p95_latency_ms:.3f} ms",
        f"  Recall@{k}: {result.recall_at_k_mean:.4f}",
        f"  MRR@{k}: {result.mrr_at_k_mean:.4f}",
    ]


def _render_plain(config: dict[str, Any], brute_result: BenchmarkResult, faiss_result: BenchmarkResult) -> str:
    k = int(config["k"])
    speedup = 0.0 if faiss_result.query_total_s == 0 else brute_result.query_total_s / faiss_result.query_total_s
    delta_recall = faiss_result.recall_at_k_mean - brute_result.recall_at_k_mean
    delta_mrr = faiss_result.mrr_at_k_mean - brute_result.mrr_at_k_mean

    lines: list[str] = []
    lines.append("=== Chapter 4 Lab: Similarity Search at Scale ===")
    lines.append(
        "Config: "
        f"N={config['n']} D={config['d']} clusters={config['clusters']} "
        f"Q={config['q']} K={config['k']} noise={config['noise']} seed={config['seed']} "
        f"index={config['index']} nlist={config['nlist']} nprobe={config['nprobe']}"
    )
    lines.append("")
    lines.extend(_line_for_result(brute_result, k=k))
    lines.append("")
    lines.extend(_line_for_result(faiss_result, k=k))
    lines.append("")
    lines.append(f"Speedup (query): {speedup:.2f}x")
    lines.append(f"Quality delta (FAISS - Brute): Recall@{k} {delta_recall:+.4f}, MRR@{k} {delta_mrr:+.4f}")

    if delta_recall < 0 or delta_mrr < 0:
        lines.append("Interpretation: ANN is faster but may reduce ranking quality; tune nprobe for tradeoffs.")
    else:
        lines.append("Interpretation: This configuration preserves quality while improving query speed.")

    return "\n".join(lines)


def _render_rich(config: dict[str, Any], brute_result: BenchmarkResult, faiss_result: BenchmarkResult) -> str:
    from rich.console import Console
    from rich.table import Table

    k = int(config["k"])
    speedup = 0.0 if faiss_result.query_total_s == 0 else brute_result.query_total_s / faiss_result.query_total_s

    console = Console(record=True, width=120)
    console.print("[bold]Chapter 4 Lab: Similarity Search at Scale[/bold]")
    console.print(
        "Config: "
        f"N={config['n']} D={config['d']} clusters={config['clusters']} "
        f"Q={config['q']} K={config['k']} noise={config['noise']} seed={config['seed']} "
        f"index={config['index']} nlist={config['nlist']} nprobe={config['nprobe']}"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Method")
    table.add_column("Build (s)")
    table.add_column("Query (s)")
    table.add_column("QPS")
    table.add_column("p50 (ms)")
    table.add_column("p95 (ms)")
    table.add_column(f"Recall@{k}")
    table.add_column(f"MRR@{k}")

    for result in (brute_result, faiss_result):
        table.add_row(
            result.name,
            f"{result.build_time_s:.4f}",
            f"{result.query_total_s:.4f}",
            f"{result.qps:.2f}",
            f"{result.p50_latency_ms:.3f}",
            f"{result.p95_latency_ms:.3f}",
            f"{result.recall_at_k_mean:.4f}",
            f"{result.mrr_at_k_mean:.4f}",
        )

    console.print(table)
    console.print(f"Speedup (query): {speedup:.2f}x")
    return console.export_text()


def render_report(config: dict[str, Any], brute_result: BenchmarkResult, faiss_result: BenchmarkResult) -> str:
    try:
        return _render_rich(config=config, brute_result=brute_result, faiss_result=faiss_result)
    except Exception:
        return _render_plain(config=config, brute_result=brute_result, faiss_result=faiss_result)
