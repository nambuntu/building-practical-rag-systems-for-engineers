from pathlib import Path
from time import perf_counter

from domain.types import IngestRecord
from app_logging import get_logger
from sources.local_source import LocalFileSource
from sources.squad_source import SquadSource
from utils.io import write_jsonl
from workflow.state import RunPaths

logger = get_logger(__name__)


def _record_to_row(record: IngestRecord) -> dict[str, object]:
    return {
        "doc_id": record.doc_id,
        "title": record.title,
        "context": record.context,
        "question": record.question,
        "answers": record.answers,
        "source_split": record.source_split,
    }


def run_ingest(
    paths: RunPaths,
    *,
    source: str,
    split: str,
    sample_size: int | None,
    input_dir: str | None,
) -> dict[str, object]:
    started_at = perf_counter()
    logger.info("Ingest stage: source=%s split=%s sample_size=%s", source, split, sample_size)
    if source == "squad":
        loader = SquadSource()
    elif source == "local":
        if not input_dir:
            raise ValueError("input_dir is required when source=local")
        loader = LocalFileSource(Path(input_dir))
    else:
        raise ValueError(f"Unsupported source '{source}'.")

    load_started = perf_counter()
    records = list(loader.load_records(split=split, sample_size=sample_size))
    logger.info("Ingest stage: loaded %s records in %.2fs", len(records), perf_counter() - load_started)

    write_started = perf_counter()
    rows = [_record_to_row(record) for record in records]
    write_jsonl(paths.ingest_raw_path, rows)
    logger.info(
        "Ingest stage: wrote artifact=%s in %.2fs (total %.2fs)",
        paths.ingest_raw_path,
        perf_counter() - write_started,
        perf_counter() - started_at,
    )

    return {
        "source": source,
        "split": split,
        "sample_size": sample_size,
        "record_count": len(rows),
        "artifact": str(paths.ingest_raw_path),
    }
