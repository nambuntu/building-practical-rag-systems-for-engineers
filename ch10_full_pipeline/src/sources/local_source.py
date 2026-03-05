from collections.abc import Iterable
from pathlib import Path

from domain.interfaces import CorpusSource
from domain.types import IngestRecord


class LocalFileSource(CorpusSource):
    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir

    def load_records(self, *, split: str, sample_size: int | None = None) -> Iterable[IngestRecord]:
        _ = split
        files = sorted(path for path in self.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
        if sample_size:
            files = files[:sample_size]

        for path in files:
            yield IngestRecord(
                doc_id=path.stem,
                title=path.stem,
                context=path.read_text(encoding="utf-8"),
                question="",
                answers=[],
                source_split="local",
            )
