from collections.abc import Iterable

from datasets import load_dataset

from domain.interfaces import CorpusSource
from domain.types import IngestRecord


class SquadSource(CorpusSource):
    def load_records(self, *, split: str, sample_size: int | None = None) -> Iterable[IngestRecord]:
        # Canonical HF usage:
        # from datasets import load_dataset
        # dataset = load_dataset("squad", split="train[:5000]")
        split_expr = split if not sample_size else f"{split}[:{sample_size}]"
        dataset = load_dataset("squad", split=split_expr)

        for row in dataset:
            answers = row.get("answers", {}).get("text", [])
            yield IngestRecord(
                doc_id=row["id"],
                title=row.get("title", ""),
                context=row.get("context", ""),
                question=row.get("question", ""),
                answers=[answer for answer in answers if answer],
                source_split=split,
            )
