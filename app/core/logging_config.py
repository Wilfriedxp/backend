"""
backend/app/core/logging_config.py
Centralised logging setup — call setup_logging() once at startup.
"""
import logging
import sys


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quieten noisy third-party loggers
    for lib in ("uvicorn.access", "multipart", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)
