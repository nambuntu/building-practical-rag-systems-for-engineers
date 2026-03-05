from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from dataset import QueryCase


@dataclass(frozen=True)
class RunResult:
    config_name: str
    chunker: str
    chunk_size: int
    overlap: int
    chunk_count: int
    avg_tokens: float
    embed_s: float
    retrieve_s: float
    recall_at_k: float
    mrr_at_k: float
    unanswerable_count: int


def render_results_table(results: list[RunResult], top_k: int) -> str:
    console = Console(record=True, width=140)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Config")
    table.add_column("#chunks", justify="right")
    table.add_column("avg_tokens", justify="right")
    table.add_column("embed_s", justify="right")
    table.add_column("retrieve_s", justify="right")
    table.add_column(f"Recall@{top_k}", justify="right")
    table.add_column(f"MRR@{top_k}", justify="right")
    table.add_column("unanswerable", justify="right")

    for result in results:
        table.add_row(
            result.config_name,
            str(result.chunk_count),
            f"{result.avg_tokens:.1f}",
            f"{result.embed_s:.4f}",
            f"{result.retrieve_s:.4f}",
            f"{result.recall_at_k:.3f}",
            f"{result.mrr_at_k:.3f}",
            str(result.unanswerable_count),
        )

    console.print(table)
    return console.export_text()


def render_interpretation(results: list[RunResult]) -> str:
    if not results:
        return ""

    best_recall = max(results, key=lambda item: item.recall_at_k)
    fastest = min(results, key=lambda item: item.retrieve_s)
    highest_mrr = max(results, key=lambda item: item.mrr_at_k)

    lines = [
        "Interpretation:",
        f"- Best recall config: {best_recall.config_name} (Recall={best_recall.recall_at_k:.3f}).",
        f"- Fastest retrieval config: {fastest.config_name} ({fastest.retrieve_s:.4f}s).",
        f"- Best ranking quality by MRR: {highest_mrr.config_name} (MRR={highest_mrr.mrr_at_k:.3f}).",
    ]
    return "\n".join(lines)


def render_query_debug(
    query: QueryCase,
    config_name: str,
    top_k: int,
    gold_chunk_id: str | None,
    phrase_locations: dict[str, list[str]],
    ranked_items: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"Debug view for {query.query_id} under {config_name}")
    lines.append(f"Query: {query.query_text}")
    lines.append(f"Notes: {query.notes}")
    lines.append(f"Required phrases: {', '.join(query.required_phrases)}")
    lines.append(f"Gold chunk: {gold_chunk_id if gold_chunk_id else 'NONE (unanswerable)'}")

    if gold_chunk_id is None:
        lines.append("Phrase locations:")
        for phrase, chunk_ids in phrase_locations.items():
            chunk_text = ", ".join(chunk_ids) if chunk_ids else "<not found>"
            lines.append(f"- {phrase}: {chunk_text}")

    lines.append(f"Top-{top_k} results:")
    for item in ranked_items:
        preview = item["text"].strip().replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        lines.append(f"- {item['chunk_id']} score={item['score']:.4f} :: {preview}")

    return "\n".join(lines)
