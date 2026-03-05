from collections import defaultdict

from domain.interfaces import EmbeddingProvider, VectorRepository
from domain.types import ContextWindow, ContextWindowRequest, SearchHit


def select_non_overlapping_hits(hits: list[SearchHit], top_k: int, window_size: int) -> list[SearchHit]:
    selected: list[SearchHit] = []

    for hit in hits:
        if len(selected) >= top_k:
            break

        is_overlap = any(
            existing.file_name == hit.file_name
            and abs(existing.sentence_number - hit.sentence_number) <= window_size
            for existing in selected
        )

        if not is_overlap:
            selected.append(hit)

    return selected


def build_window_requests(hits: list[SearchHit], window_size: int) -> list[ContextWindowRequest]:
    ranges_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for hit in hits:
        start = max(1, hit.sentence_number - window_size)
        end = hit.sentence_number + window_size
        ranges_by_file[hit.file_name].append((start, end))

    merged_requests: list[ContextWindowRequest] = []
    for file_name, ranges in ranges_by_file.items():
        ranges.sort(key=lambda item: item[0])
        merged: list[list[int]] = []

        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        merged_requests.extend(
            ContextWindowRequest(
                file_name=file_name,
                start_sentence_number=start,
                end_sentence_number=end,
            )
            for start, end in merged
        )

    merged_requests.sort(key=lambda item: (item.file_name, item.start_sentence_number))
    return merged_requests


class RetrievalService:
    def __init__(self, embedding_provider: EmbeddingProvider, repository: VectorRepository) -> None:
        self.embedding_provider = embedding_provider
        self.repository = repository

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int,
        context_window: int,
    ) -> list[ContextWindow]:
        query_embedding = self.embedding_provider.embed_query(query)
        search_limit = top_k * (2 * context_window + 1)
        hits = self.repository.search(query_vector=query_embedding, limit=search_limit)
        selected_hits = select_non_overlapping_hits(hits=hits, top_k=top_k, window_size=context_window)

        if not selected_hits:
            return []

        requests = build_window_requests(hits=selected_hits, window_size=context_window)
        return self.repository.fetch_context_windows(requests)
