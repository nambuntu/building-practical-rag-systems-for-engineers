"""Canonical entrypoint for the stage-based RAG learning pipeline."""

import netrc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app_logging import setup_logging

setup_logging(os.getenv("RAG_LOG_LEVEL", "INFO"))


def _load_hf_token_from_netrc() -> None:
    if os.getenv("HF_TOKEN"):
        return
    try:
        netrc_data = netrc.netrc()
        auth = netrc_data.authenticators("huggingface.co") or netrc_data.authenticators("hf.co")
    except (FileNotFoundError, netrc.NetrcParseError):
        return
    if not auth:
        return
    login, _, password = auth
    if login == "__token__" and password:
        os.environ["HF_TOKEN"] = password


_load_hf_token_from_netrc()

from workflow.app import app


if __name__ == "__main__":
    app()
