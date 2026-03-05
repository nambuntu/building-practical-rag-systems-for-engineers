from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.io import read_json, write_json


@dataclass(slots=True)
class RunPaths:
    run_id: str
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def ingest_raw_path(self) -> Path:
        return self.root / "ingest" / "raw.jsonl"

    @property
    def prepare_chunks_path(self) -> Path:
        return self.root / "prepare" / "chunks.jsonl"

    @property
    def prepare_eval_path(self) -> Path:
        return self.root / "prepare" / "eval_examples.jsonl"

    @property
    def index_manifest_path(self) -> Path:
        return self.root / "index" / "index_manifest.json"

    @property
    def query_results_path(self) -> Path:
        return self.root / "query" / "results.jsonl"

    @property
    def eval_report_path(self) -> Path:
        return self.root / "evaluate" / "report.json"

    @property
    def run_log_path(self) -> Path:
        return self.root / "run.log"


def build_run_paths(runs_dir: Path, run_id: str) -> RunPaths:
    return RunPaths(run_id=run_id, root=runs_dir / run_id)


def ensure_run_dirs(paths: RunPaths) -> None:
    for folder in ["ingest", "prepare", "index", "query", "evaluate"]:
        (paths.root / folder).mkdir(parents=True, exist_ok=True)


def load_manifest(paths: RunPaths) -> dict[str, Any]:
    if paths.manifest_path.exists():
        return read_json(paths.manifest_path)
    return {
        "run_id": paths.run_id,
        "stages": {},
    }


def save_manifest(paths: RunPaths, manifest: dict[str, Any]) -> None:
    write_json(paths.manifest_path, manifest)
