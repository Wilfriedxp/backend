"""
backend/app/api/v1/endpoints/reports.py

POST /api/v1/reports/generate          build HTML report
POST /api/v1/reports/send-email        send to logged-in user's email (or custom address)
GET  /api/v1/reports/smtp-status       show what SMTP settings are loaded (no password)
POST /api/v1/reports/test-smtp         step-by-step SMTP connection test
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.endpoints.auth import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.app_user import AppUser
from app.schemas.models import (
    GenerateReportResponse,
    SendReportEmailRequest,
    SendReportEmailResponse,
)
from app.services.report_service import (
    generate_report_html,
    send_report_email,
    test_smtp_connection,
)
from sqlalchemy.orm import Session

router = APIRouter(prefix="/reports", tags=["Reports"])
log    = logging.getLogger("endpoint.reports")


# ── Generate ──────────────────────────────────────────────────────────────────
@router.post(
    "/generate",
    response_model=GenerateReportResponse,
    summary="Generate a BI HTML report",
)
async def generate_report(
    current_user: AppUser = Depends(get_current_user),
) -> GenerateReportResponse:
    try:
        html = generate_report_html()
    except Exception as exc:
        log.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report error: {exc}")

    # Auto-send email if user has enabled it in their profile settings
    auto_sent    = False
    auto_sent_to = None
    if current_user.auto_email_reports:
        try:
            send_report_email(
                to_email = current_user.email,
                subject  = "WebMine BI — Automated Report",
            )
            auto_sent    = True
            auto_sent_to = current_user.email
            log.info("Auto-email sent to %s", current_user.email)
        except Exception as exc:
            log.warning("Auto-email failed: %s", exc)

    return GenerateReportResponse(
        message=(
            f"Report generated and automatically emailed to {auto_sent_to}."
            if auto_sent else
            f"Report generated for {current_user.full_name}."
        ),
        report_html=html,
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        kpi_count=html.count("<tr>"),
    )


# ── Send email ────────────────────────────────────────────────────────────────
@router.post(
    "/send-email",
    response_model=SendReportEmailResponse,
    summary="Send BI report to an email address",
    description=(
        "Generates the HTML report and sends it via SMTP.  "
        "**to_email is optional** — when omitted the report goes to the "
        "logged-in user's own registered email automatically.  "
        "Supply a different address to forward to a supervisor.  "
        "SMTP must be configured in `backend/.env`."
    ),
)
async def send_report(
    request:      SendReportEmailRequest,
    current_user: AppUser  = Depends(get_current_user),
    db:           Session  = Depends(get_db),
) -> SendReportEmailResponse:
    # ── Resolve recipient ──────────────────────────────────────────────────────
    # If the frontend did not supply to_email (or left it None / empty),
    # default to the logged-in user's own registered email.
    if request.to_email:
        to_email = str(request.to_email)
    else:
        to_email = current_user.email   # ← dynamic, from database, never from .env

    try:
        result = send_report_email(
            to_email = to_email,
            subject  = request.subject,
            cc_email = str(request.cc_email) if request.cc_email else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Return the full SMTP diagnostic message — not a generic one
        log.exception("Email send failed → %s", to_email)
        raise HTTPException(status_code=502, detail=str(exc))

    return SendReportEmailResponse(**result)


# ── SMTP status (safe — never shows password) ─────────────────────────────────
@router.get(
    "/smtp-status",
    summary="Show which SMTP settings are currently loaded",
    description=(
        "Returns the SMTP configuration visible to the app — without the password.  "
        "Use this to verify that your .env file was read correctly."
    ),
)
async def smtp_status(
    current_user: AppUser = Depends(get_current_user),
) -> dict:
    import os
    env_path   = settings.env_file_location()
    env_exists = os.path.isfile(env_path)
    return {
        "env_file_path":   env_path,
        "env_file_exists": env_exists,
        "smtp_host":       settings.SMTP_HOST    or "(not set)",
        "smtp_port":       settings.SMTP_PORT,
        "smtp_user":       settings.SMTP_USER    or "(not set)",
        "password_set":    bool(settings.SMTP_PASSWORD),
        "email_from":      settings.EMAIL_FROM,
        "diagnosis": (
            "OK — .env loaded and SMTP settings present"
            if env_exists and settings.SMTP_HOST else
            f"PROBLEM — .env not found at: {env_path}"
            if not env_exists else
            "PROBLEM — .env exists but SMTP_HOST is empty — check .env contents"
        ),
    }


# ── Step-by-step SMTP test ────────────────────────────────────────────────────
@router.post(
    "/test-smtp",
    summary="Run a step-by-step SMTP connection test",
    description=(
        "Tests the SMTP connection in five independent stages and returns "
        "a detailed diagnostic report.  Call this when email sending fails "
        "to find exactly where the problem is."
    ),
)
async def test_smtp(
    current_user: AppUser = Depends(get_current_user),
) -> dict:
    """
    Returns a dict with:
    - settings_loaded  — are SMTP_HOST / USER / PASSWORD set in .env?
    - smtp_host, smtp_port, smtp_user, password_set
    - stages — {stage_name: "✓ …" or "✗ …"}
    - ready  — True only when all stages pass (email will work)
    - error  — plain-English explanation of the first failure
    """
    diag = test_smtp_connection()
    log.info("SMTP test result: ready=%s  error=%s", diag["ready"], diag.get("error"))
    return diag
