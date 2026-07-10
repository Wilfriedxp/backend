"""
backend/app/main.py
FastAPI application factory for WebMine BI.

Run:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs   ← Swagger UI
    http://localhost:8000/redoc  ← ReDoc
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging(debug=settings.DEBUG)
log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and pre-load trained models from disk at startup."""
    from app.db.init_db import init_db
    from app.state import app_state

    # Create all tables (idempotent — safe to run on every startup)
    init_db()
    log.info("✓ Database tables ready.")

    # Add ml/ to sys.path so training modules are importable
    ml_dir = str(settings.ML_DIR)
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)

    if settings.return_model_path.exists():
        try:
            from training.train_return_classifier import ReturnUserClassifier
            app_state.return_classifier = ReturnUserClassifier.load(
                settings.return_model_path
            )
            log.info("✓ Return-user model loaded from disk.")
        except Exception as exc:
            log.warning("Could not pre-load return model: %s", exc)

    if settings.traffic_model_path.exists():
        try:
            from training.train_traffic_forecaster import TrafficForecaster
            app_state.traffic_forecaster = TrafficForecaster.load(
                settings.traffic_model_path
            )
            log.info("✓ Traffic forecast model loaded from disk.")
        except Exception as exc:
            log.warning("Could not pre-load traffic model: %s", exc)

    log.info("═══ %s v%s ready ═══", settings.APP_NAME, settings.APP_VERSION)
    yield
    log.info("═══ API shutting down ═══")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = settings.APP_NAME,
    description = (
        "RESTful API for the **User Web Access Records Mining for Business "
        "Intelligence** Final Year Project.\n\n"
        "**Workflow:**\n"
        "1. `POST /api/v1/upload` — upload web access log CSV\n"
        "2. `POST /api/v1/train-return-model` — train the RF classifier\n"
        "3. `POST /api/v1/train-traffic-model` — train the RF regressor\n"
        "4. `GET  /api/v1/dashboard-data` — fetch KPIs + chart data\n"
        "5. `POST /api/v1/predict-return` — predict user return likelihood\n"
        "6. `POST /api/v1/predict-traffic` — predict tomorrow's visitors\n"
        "7. `POST /api/v1/reports/send-email` — email the BI report\n"
    ),
    version     = settings.APP_VERSION,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# CORS:
# - settings.CORS_ORIGINS  → React frontend (localhost:3000 / localhost:5173)
# - allow_origin_regex      → Chrome extension (any ID: chrome-extension://xxxx…)
#   We cannot hardcode the extension ID because it changes per Chrome installation.
#   The /collect endpoint authenticates via X-Collector-Token so open CORS is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins      = settings.CORS_ORIGINS,
    allow_origin_regex = r"chrome-extension://.*",   # ← fixes extension CORS
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*", "X-Collector-Token"],
    expose_headers     = ["X-Process-Time-Ms"],
)


# ── Request timing ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(ms)
    return response


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ── Mount routes ───────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


# ── Health & root ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["Health"])
async def health():
    from app.state import app_state
    return {
        "status":               "healthy",
        "data_loaded":          app_state.raw_df is not None,
        "return_model_ready":   app_state.return_classifier is not None,
        "traffic_model_ready":  app_state.traffic_forecaster is not None,
        "smtp_configured":      bool(settings.SMTP_HOST and settings.SMTP_USER),
    }
