"""
backend/app/state.py
Global in-memory application state shared across all requests.
In production, migrate this to a proper database session + cache layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class AppState:
    # ── Uploaded / processed data ──────────────────────────────────────────────
    raw_df:        Optional[pd.DataFrame] = None   # raw access log
    features_df:   Optional[pd.DataFrame] = None   # per-user feature matrix
    daily_traffic: Optional[pd.DataFrame] = None   # date → daily visit count

    # ── Trained ML models (in-memory handles) ─────────────────────────────────
    return_classifier:  Optional[Any] = None
    traffic_forecaster: Optional[Any] = None

    # ── Metadata ───────────────────────────────────────────────────────────────
    upload_stats:    Dict = field(default_factory=dict)
    return_metrics:  Dict = field(default_factory=dict)
    traffic_metrics: Dict = field(default_factory=dict)
    prediction_log:  list = field(default_factory=list)
    report_log:      list = field(default_factory=list)


# Module-level singleton — import this everywhere
app_state = AppState()
