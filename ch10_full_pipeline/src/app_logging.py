import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level.upper(), format=_LOG_FORMAT)
        return
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging("INFO")
    return logging.getLogger(name)


def attach_file_logger(path: Path, level: str = "INFO") -> None:
    root = logging.getLogger()
    desired = str(path.resolve())

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == desired:
                return

    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(desired, encoding="utf-8")
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)
