"""
backend/app/api/v1/endpoints/collect.py
POST /api/v1/collect
Receives page-view / click / session events from the Chrome extension
running on the user's own website.

Authentication: X-Collector-Token header (from profile page, not a JWT).
This keeps the extension from needing to store or expose JWT tokens.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.app_user import AppUser
from app.state import app_state

router = APIRouter(prefix="/collect", tags=["Chrome Extension"])
log    = logging.getLogger("endpoint.collect")


# ── Event schemas ─────────────────────────────────────────────────────────────
class PageEvent(BaseModel):
    event_type:   str            # "pageview" | "click" | "session_start" | "session_end"
    url:          str
    referrer:     Optional[str]  = None
    user_agent:   Optional[str]  = None
    session_id:   Optional[str]  = None
    duration_sec: Optional[int]  = None   # for session_end events
    timestamp:    Optional[str]  = None   # ISO string; server time used if absent


class CollectRequest(BaseModel):
    events: List[PageEvent] = Field(..., min_length=1, max_length=100)


class CollectResponse(BaseModel):
    received:  int
    status:    str = "ok"
    timestamp: str


# ── Token dependency ──────────────────────────────────────────────────────────
def get_user_by_collector_token(
    x_collector_token: str = Header(..., description="User's collector token from profile page"),
    db: Session = Depends(get_db),
) -> AppUser:
    user = db.query(AppUser).filter(
        AppUser.collector_token == x_collector_token,
        AppUser.is_active       == True,
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid collector token. "
                "Find your token on the Profile page of the WebMine BI dashboard."
            ),
        )
    return user


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("", response_model=CollectResponse, summary="Receive events from Chrome extension")
async def collect_events(
    payload: CollectRequest,
    user:    AppUser = Depends(get_user_by_collector_token),
    db:      Session = Depends(get_db),
) -> CollectResponse:
    """
    Receives batched page events from the WebMine BI Chrome extension
    running on the user's own online platform / website.

    Each event becomes a row that the preprocessing pipeline can treat
    as equivalent to a server-side access log row.

    The endpoint converts events to the same format as uploaded CSV logs
    and appends them to the in-memory raw_df so the dashboard updates
    in real time without a re-upload.
    """
    import pandas as pd
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for ev in payload.events:
        # Only process actual page views for access log equivalence
        if ev.event_type not in ("pageview", "session_start"):
            continue

        ts = ev.timestamp or now_str
        # Normalise ISO → space-separated
        ts = ts.replace("T", " ").replace("Z", "").split(".")[0]

        rows.append({
            "ip_address":  ev.session_id or "ext-unknown",
            "timestamp":   ts,
            "method":      "GET",
            "url":         ev.url,
            "status_code": 200,
            "bytes_sent":  0,
            "user_agent":  ev.user_agent or "WebMine-Extension/1.0",
            "referrer":    ev.referrer or "-",
        })

    if rows:
        new_df = pd.DataFrame(rows)
        if app_state.raw_df is not None:
            app_state.raw_df = pd.concat(
                [app_state.raw_df, new_df], ignore_index=True
            )
        else:
            app_state.raw_df = new_df

    log.info(
        "Extension collect: user=%s  events=%d  pageviews=%d",
        user.email, len(payload.events), len(rows),
    )
    return CollectResponse(
        received  = len(payload.events),
        timestamp = now_str,
    )


# ── Status endpoint (extension health check) ─────────────────────────────────
@router.get("/status", summary="Extension connectivity check")
async def collect_status(
    user: AppUser = Depends(get_user_by_collector_token),
) -> dict:
    """
    Called by the extension on startup to verify the token is valid
    and the backend is reachable.
    """
    return {
        "status":    "connected",
        "user":      user.full_name,
        "email":     user.email,
        "timestamp": datetime.utcnow().isoformat(),
    }
