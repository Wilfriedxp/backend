"""
preprocessing_pipeline.py
==========================
Pandas preprocessing pipeline for CSV web access logs.

Project path : backend/app/ml/pipelines/preprocessing_pipeline.py

Input  : Raw CSV web access log (Apache / Nginx combined-log format or similar)
Output : Per-user feature DataFrame with five engineered features:

    session_duration   — mean session length in minutes
    page_views         — total page views across all sessions
    visit_frequency    — number of distinct sessions
    bounce_rate        — fraction of single-page-view sessions  [0–1]
    navigation_depth   — mean URL path depth across all page views

Usage
-----
    from backend.app.ml.pipelines.preprocessing_pipeline import (
        WebLogPreprocessor, PipelineConfig
    )
    features = WebLogPreprocessor().run("path/to/access.csv")

Required CSV columns
--------------------
    ip_address, timestamp, url, status_code
Optional:
    method, bytes_sent, referrer, user_agent
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("preprocessing_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """All tunable knobs for the preprocessing pipeline.

    Attributes
    ----------
    session_timeout_min : int
        Inactivity gap (minutes) that marks the start of a new session.
    min_url_length : int
        Drop URLs shorter than this many characters.
    filter_static_assets : bool
        Remove requests for images, CSS, JS, fonts, etc.
    filter_bots : bool
        Remove rows whose user_agent matches known crawler patterns.
    valid_status_codes : list[int]
        Retain only requests with these HTTP status codes.
    timestamp_formats : list[str]
        strptime-style formats tried in order during timestamp parsing.
    """
    session_timeout_min: int   = 30
    min_url_length: int        = 1
    filter_static_assets: bool = True
    filter_bots: bool          = True
    valid_status_codes: List[int] = field(
        default_factory=lambda: [200, 201, 206, 301, 302, 304]
    )
    timestamp_formats: List[str] = field(
        default_factory=lambda: [
            "%d/%b/%Y:%H:%M:%S %z",   # Apache:  10/Oct/2024:13:55:36 +0000
            "%Y-%m-%d %H:%M:%S",      # ISO-ish: 2024-10-10 13:55:36
            "%Y-%m-%dT%H:%M:%S",      # ISO 8601
            "%d/%m/%Y %H:%M:%S",      # EU date
            "%m/%d/%Y %H:%M:%S",      # US date
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns  (module-level for performance)
# ─────────────────────────────────────────────────────────────────────────────
_STATIC_EXT_PAT = (
    r"\.(?:css|js|jsx|ts|tsx|jpg|jpeg|png|gif|ico|svg|webp"
    r"|woff2?|ttf|eot|mp4|webm|ogg|mp3|pdf|xml|map|txt)(?:\?.*)?$"
)
_BOT_UA_PAT = (
    r"(?:bot|crawl|spider|slurp|bingpreview|mediapartners|googlefavicon"
    r"|facebookexternalhit|twitterbot|linkedinbot|pingdom|uptimerobot"
    r"|nagios|python-urllib|python-requests|go-http-client"
    r"|curl|wget|axios|libwww-perl|scrapy|mechanize|ahrefs|semrush)"
)
_REQUIRED_COLUMNS: set[str] = {"ip_address", "timestamp", "url", "status_code"}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────
class WebLogPreprocessor:
    """Five-stage preprocessing pipeline for web access logs.

    Stages
    ------
    1. load_and_validate   — CSV ingestion + column check
    2. parse_timestamps    — multi-format timestamp parsing, UTC normalisation
    3. clean               — status filtering, URL normalisation, bot/static removal
    4. reconstruct_sessions — inactivity-gap session labelling
    5. engineer_features   — session → user aggregation into 5 ML-ready features

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  Defaults to ``PipelineConfig()`` if omitted.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.cfg = config or PipelineConfig()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    def load_and_validate(self, filepath: str | Path) -> pd.DataFrame:
        """Load CSV and verify all required columns are present."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        log.info("Loading '%s' …", path.name)
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}\n"
                f"Found: {sorted(df.columns)}"
            )
        log.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
        return df

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect timestamp format and parse to naive UTC datetime64."""
        df = df.copy()
        raw = df["timestamp"].astype(str).str.strip()
        parsed: Optional[pd.Series] = None

        for fmt in self.cfg.timestamp_formats:
            candidate = pd.to_datetime(raw, format=fmt, errors="coerce")
            if candidate.notna().mean() >= 0.80:
                parsed = candidate
                log.info(
                    "Timestamp format '%s' matched %.0f%% of rows.",
                    fmt, candidate.notna().mean() * 100,
                )
                break

        if parsed is None:
            parsed = pd.to_datetime(raw, format="mixed", errors="coerce")
            log.warning(
                "Mixed-format inference; %.0f%% rows parsed.",
                parsed.notna().mean() * 100,
            )

        if hasattr(parsed, "dt") and parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

        df["timestamp"] = parsed
        n_bad = df["timestamp"].isna().sum()
        if n_bad:
            log.warning("Dropping %d rows with unparseable timestamps.", n_bad)
            df = df.dropna(subset=["timestamp"])

        return df.reset_index(drop=True)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning rules; return a pruned DataFrame."""
        df = df.copy()
        before = len(df)

        # Status codes
        df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce")
        df = df.dropna(subset=["status_code"])
        df["status_code"] = df["status_code"].astype(int)
        df = df[df["status_code"].isin(self.cfg.valid_status_codes)]

        # URL presence and normalisation
        df["url"] = df["url"].astype(str).str.strip()
        df = df[df["url"].str.len() >= max(self.cfg.min_url_length, 1)]
        df["url_path"] = df["url"].str.extract(r"^([^?#]*)", expand=False)
        df["url_path"] = df["url_path"].str.lower().str.rstrip("/")
        df["url_path"] = df["url_path"].where(df["url_path"] != "", other="/")

        # Static assets
        if self.cfg.filter_static_assets:
            is_static = df["url_path"].str.contains(
                _STATIC_EXT_PAT, flags=re.IGNORECASE, regex=True, na=False
            )
            df = df[~is_static]
            log.info("Removed %d static-asset requests.", is_static.sum())

        # Bot user-agents
        if self.cfg.filter_bots and "user_agent" in df.columns:
            df = df.copy()
            df["user_agent"] = df["user_agent"].fillna("").astype(str)
            is_bot = df["user_agent"].str.contains(
                _BOT_UA_PAT, flags=re.IGNORECASE, regex=True, na=False
            )
            df = df[~is_bot]
            log.info("Removed %d bot/crawler requests.", is_bot.sum())

        df = df.dropna(subset=["ip_address", "timestamp"]).copy()
        df["ip_address"] = df["ip_address"].astype(str).str.strip()

        after = len(df)
        log.info(
            "Cleaning: %d → %d rows (removed %d / %.1f%%).",
            before, after, before - after, 100.0 * (before - after) / max(before, 1),
        )
        return df.reset_index(drop=True)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    def reconstruct_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign session IDs using the inactivity-gap method.

        New session when gap since user's last request > session_timeout_min.
        Adds: session_id (str), session_seq (1-based request rank in session).
        """
        df = df.copy()
        df = df.sort_values(["ip_address", "timestamp"]).reset_index(drop=True)
        timeout = pd.Timedelta(minutes=self.cfg.session_timeout_min)

        df["_delta"]    = df.groupby("ip_address")["timestamp"].diff()
        df["_new_sess"] = df["_delta"].isna() | (df["_delta"] > timeout)
        df["_sess_ctr"] = (
            df.groupby("ip_address")["_new_sess"].cumsum().astype(int)
        )
        df["session_id"]  = df["ip_address"] + "_s" + df["_sess_ctr"].astype(str)
        df["session_seq"] = df.groupby("session_id").cumcount() + 1
        df = df.drop(columns=["_delta", "_new_sess", "_sess_ctr"])

        log.info(
            "Sessions: %d sessions across %d users (avg %.1f/user).",
            df["session_id"].nunique(),
            df["ip_address"].nunique(),
            df["session_id"].nunique() / max(df["ip_address"].nunique(), 1),
        )
        return df

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate session data into a per-user feature matrix.

        Features
        --------
        session_duration  : mean session duration in minutes
        page_views        : total page views across all sessions
        visit_frequency   : number of distinct sessions
        bounce_rate       : share of single-page sessions  [0–1]
        navigation_depth  : mean URL depth (path segment count)
        """
        df = df.copy()

        # URL depth per request
        df["_url_depth"] = (
            df["url_path"].str.strip("/").str.split("/")
            .apply(lambda parts: sum(1 for p in parts if p))
        )

        # Session-level aggregation
        sess = (
            df.groupby("session_id", sort=False)
            .agg(
                ip_address   =("ip_address",  "first"),
                sess_start   =("timestamp",   "min"),
                sess_end     =("timestamp",   "max"),
                _page_views  =("url_path",    "count"),
                _mean_depth  =("_url_depth",  "mean"),
            )
            .reset_index()
        )
        sess["duration_min"] = (
            (sess["sess_end"] - sess["sess_start"])
            .dt.total_seconds().clip(lower=0).div(60.0)
        )
        sess["_is_bounce"] = (sess["_page_views"] == 1).astype(int)

        # User-level aggregation
        features = (
            sess.groupby("ip_address", sort=False)
            .agg(
                visit_frequency  =("session_id",   "count"),
                page_views       =("_page_views",  "sum"),
                session_duration =("duration_min", "mean"),
                _bounce_n        =("_is_bounce",   "sum"),
                navigation_depth =("_mean_depth",  "mean"),
                first_visit      =("sess_start",   "min"),
                last_visit       =("sess_end",     "max"),
            )
            .reset_index()
            .rename(columns={"ip_address": "user_id"})
        )

        features["bounce_rate"] = (
            features["_bounce_n"] / features["visit_frequency"]
        ).clip(0.0, 1.0)
        features = features.drop(columns=["_bounce_n"])

        features["page_views"]       = features["page_views"].astype(int)
        features["visit_frequency"]  = features["visit_frequency"].astype(int)
        features["session_duration"] = features["session_duration"].round(2)
        features["bounce_rate"]      = features["bounce_rate"].round(4)
        features["navigation_depth"] = features["navigation_depth"].round(2)

        col_order = [
            "user_id", "session_duration", "page_views",
            "visit_frequency", "bounce_rate", "navigation_depth",
            "first_visit", "last_visit",
        ]
        features = (
            features[col_order]
            .sort_values("visit_frequency", ascending=False)
            .reset_index(drop=True)
        )
        log.info("Features ready: %d users × 5 ML features.", len(features))
        return features

    # ── Orchestrator ───────────────────────────────────────────────────────────
    def run(self, filepath: str | Path) -> pd.DataFrame:
        """Run all five stages end-to-end.

        Returns
        -------
        pd.DataFrame
            Per-user feature matrix (one row per unique IP address).
        """
        log.info("── Preprocessing pipeline start ─────────────────────────")
        df = self.load_and_validate(filepath)
        df = self.parse_timestamps(df)
        df = self.clean(df)
        df = self.reconstruct_sessions(df)
        features = self.engineer_features(df)
        log.info("── Preprocessing pipeline complete ──────────────────────")
        return features


# ─────────────────────────────────────────────────────────────────────────────
# Sample-data generator  (used by the training script)
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample_log(
    n_days: int = 60,
    n_users: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic web access log spanning ``n_days`` days.

    Produces a realistic mix of:
    - Loyal users  (20 %) — high frequency, spread across all days
    - Regular users (25 %) — moderate frequency, mostly in the first 75 % of days
    - Casual users  (35 %) — low frequency, only in the first two-thirds
    - One-time users (20 %) — single visit, only in the first half

    This distribution naturally yields ~30–40 % positive labels when the
    training script applies a temporal window to create will_return targets.

    Parameters
    ----------
    n_days  : int  — span of the log in days  (default 60)
    n_users : int  — number of unique simulated IP addresses  (default 200)
    seed    : int  — random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01 00:00:00")

    hour_weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.8, 1.5, 2.5, 3.5, 4.0, 4.5, 4.0,
        3.5, 4.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.7,
    ])
    hour_weights /= hour_weights.sum()

    pages = [
        "/", "/home", "/about", "/products", "/products/item-1",
        "/products/item-2", "/products/item-3", "/blog", "/blog/post-1",
        "/blog/post-2", "/contact", "/pricing", "/cart", "/checkout",
        "/dashboard", "/dashboard/analytics",
    ]
    static = ["/static/style.css", "/assets/logo.png", "/favicon.ico"]
    real_uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605",
        "Mozilla/5.0 (Linux; Android 13) Chrome Mobile",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile Safari",
    ]
    bot_uas = ["Googlebot/2.1", "python-requests/2.31.0"]

    train_cutoff  = int(n_days * 0.75)   # 45 days for n_days=60
    casual_cutoff = int(n_days * 0.67)   # ~40 days
    one_time_max  = int(n_days * 0.50)   # first 30 days

    tiers = rng.choice(
        ["loyal", "regular", "casual", "one_time"],
        size=n_users,
        p=[0.20, 0.25, 0.35, 0.20],
    )

    records: list[dict] = []

    for uid, tier in enumerate(tiers):
        ip = f"10.{(uid // 254) + 1}.{(uid % 254) + 1}.1"

        if tier == "loyal":
            n_visit_days  = int(rng.integers(10, 22))
            avail_days    = np.arange(n_days)
            pages_per_day = (5, 14)
        elif tier == "regular":
            n_visit_days  = int(rng.integers(4, 10))
            avail_days    = np.arange(min(train_cutoff + 10, n_days))
            pages_per_day = (2, 7)
        elif tier == "casual":
            n_visit_days  = int(rng.integers(2, 5))
            avail_days    = np.arange(casual_cutoff)
            pages_per_day = (1, 4)
        else:  # one_time
            n_visit_days  = 1
            avail_days    = np.arange(one_time_max)
            pages_per_day = (1, 3)

        n_visit_days = min(n_visit_days, len(avail_days))
        visit_days   = sorted(
            rng.choice(avail_days, size=n_visit_days, replace=False).tolist()
        )

        # Regular users: 55 % chance of a return visit in prediction window
        if tier == "regular" and rng.random() < 0.55:
            pred_window = np.arange(train_cutoff, n_days)
            if len(pred_window):
                extra = int(rng.integers(1, min(4, len(pred_window) + 1)))
                extra_days = rng.choice(pred_window, size=extra, replace=False).tolist()
                visit_days = sorted(set(visit_days) | set(extra_days))

        for day in visit_days:
            n_pages   = int(rng.integers(*pages_per_day))
            visit_hour = int(rng.choice(np.arange(24), p=hour_weights))
            for seq in range(n_pages):
                ts = (
                    base
                    + pd.Timedelta(days=int(day))
                    + pd.Timedelta(hours=visit_hour)
                    + pd.Timedelta(minutes=int(rng.integers(0, 60)))
                    + pd.Timedelta(seconds=seq * 45 + int(rng.integers(0, 30)))
                )
                records.append({
                    "ip_address":  ip,
                    "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "method":      rng.choice(["GET", "POST"], p=[0.92, 0.08]),
                    "url":         str(rng.choice(pages)),
                    "status_code": int(rng.choice([200, 301, 304], p=[0.87, 0.07, 0.06])),
                    "bytes_sent":  int(rng.integers(400, 60_000)),
                    "user_agent":  str(rng.choice(real_uas)),
                })

        # Sprinkle static-asset requests (to be filtered out)
        if rng.random() < 0.35 and visit_days:
            ts = base + pd.Timedelta(days=int(visit_days[0]))
            records.append({
                "ip_address":  ip,
                "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                "method":      "GET",
                "url":         str(rng.choice(static)),
                "status_code": 200,
                "bytes_sent":  int(rng.integers(1_000, 100_000)),
                "user_agent":  str(rng.choice(real_uas)),
            })

    # Inject bot traffic (to be filtered out)
    for _ in range(80):
        ts = base + pd.Timedelta(days=int(rng.integers(0, n_days)))
        records.append({
            "ip_address":  f"203.0.{int(rng.integers(1, 254))}.{int(rng.integers(1, 254))}",
            "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
            "method":      "GET",
            "url":         str(rng.choice(pages)),
            "status_code": 200,
            "bytes_sent":  1024,
            "user_agent":  str(rng.choice(bot_uas)),
        })

    return pd.DataFrame(records)
