"""
upload.py  —  backend/app/api/v1/endpoints/upload.py
Handles CSV file ingestion and triggers the preprocessing pipeline.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.models import UploadResponse
from app.services.ml_service import preprocess_uploaded_log
from app.core.config import settings

router = APIRouter()
log    = logging.getLogger("endpoint.upload")

REQUIRED_COLS = {"ip_address", "timestamp", "url", "status_code"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload web access log CSV",
    description=(
        "Accepts a CSV web access log file, runs the preprocessing pipeline, "
        "and stores the per-user feature matrix in memory for subsequent training calls."
    ),
)
async def upload_log(file: UploadFile = File(...)) -> UploadResponse:
    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only CSV files are accepted.",
        )

    # ── Read content ──────────────────────────────────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit ({size_mb:.1f} MB uploaded).",
        )

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

    raw_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_df.columns]
    missing = REQUIRED_COLS - set(raw_df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    log.info("Received '%s' — %d rows, %.2f MB", file.filename, len(raw_df), size_mb)

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        stats = preprocess_uploaded_log(raw_df)
    except Exception as exc:
        log.exception("Preprocessing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing error: {exc}",
        )

    # ── Build response ────────────────────────────────────────────────────────
    from app.state import app_state
    preview = (
        app_state.features_df
        .head(5)
        .drop(columns=["first_visit", "last_visit"], errors="ignore")
        .to_dict(orient="records")
    )

    return UploadResponse(
        message=f"Successfully processed '{file.filename}'",
        rows_ingested=stats["total_rows"],
        users_found=stats["unique_users"],
        date_range={
            "start": stats["date_range_start"],
            "end":   stats["date_range_end"],
        },
        feature_preview=preview,
        upload_stats=stats,
    )
