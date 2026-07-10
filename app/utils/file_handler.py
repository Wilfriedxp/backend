"""
backend/app/utils/file_handler.py
CSV validation helpers used by the upload endpoint.
"""
from __future__ import annotations
import io
import logging
from typing import Set, Tuple
import pandas as pd

log = logging.getLogger("file_handler")

REQUIRED_COLS: Set[str] = {"ip_address", "timestamp", "url", "status_code"}
MAX_MB = 50


def validate_and_parse_csv(content: bytes) -> Tuple[pd.DataFrame, dict]:
    """Parse and validate a CSV byte payload.

    Returns
    -------
    (df, stats) where stats contains row count, columns, size_mb.

    Raises
    ------
    ValueError  if file is empty, unparseable, or missing required columns.
    """
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise ValueError(f"File size {size_mb:.1f} MB exceeds {MAX_MB} MB limit.")
    if len(content) == 0:
        raise ValueError("Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    if len(df) == 0:
        raise ValueError("CSV has no data rows.")

    stats = {
        "rows":     len(df),
        "columns":  list(df.columns),
        "size_mb":  round(size_mb, 3),
    }
    log.info("CSV validated: %d rows, %.2f MB", len(df), size_mb)
    return df, stats
