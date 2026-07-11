"""
backend/app/services/report_service.py
HTML report generation + SMTP email delivery.

TWO email addresses are involved — they are completely different:
  SMTP_USER  (in .env)   = the Gmail account that SENDS the email  (your app account)
  to_email   (from DB)   = the logged-in user's registered email    (who RECEIVES it)

These must never be confused.  SMTP_USER is fixed infrastructure.
to_email is dynamic and comes from the authenticated user's database record.
"""
from __future__ import annotations

import logging
import smtplib
import socket
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

from app.core.config import settings
from app.services.dashboard_service import build_dashboard_payload

log = logging.getLogger("report_service")


# ─────────────────────────────────────────────────────────────────────────────
# SMTP Diagnostic — call this first to find exactly what is broken
# ─────────────────────────────────────────────────────────────────────────────
def test_smtp_connection() -> Dict:
    """
    Run a step-by-step SMTP connection test and return a detailed
    diagnostic dictionary.  Each stage is tested independently so
    the caller can see exactly where the failure occurs.

    Stages tested
    -------------
    1. settings_loaded  — are SMTP_HOST / SMTP_USER / SMTP_PASSWORD set?
    2. tcp_connect      — can we open a TCP socket to host:port?
    3. ehlo             — does the server accept our EHLO greeting?
    4. starttls         — can we upgrade to TLS?
    5. login            — do the credentials authenticate successfully?

    Returns
    -------
    dict with keys:
        settings_loaded (bool)
        smtp_host, smtp_port, smtp_user, password_set
        stages  — dict of stage_name → "✓ …" or "✗ …"
        ready   — True only when all five stages pass
        error   — plain-English explanation of the first failure found
    """
    result: Dict = {
        "settings_loaded": False,
        "smtp_host":       settings.SMTP_HOST   or "(not set — check .env)",
        "smtp_port":       settings.SMTP_PORT,
        "smtp_user":       settings.SMTP_USER   or "(not set — check .env)",
        "password_set":    bool(settings.SMTP_PASSWORD),
        "stages":          {},
        "ready":           False,
        "error":           None,
    }

    # ── Stage 1: settings present ──────────────────────────────────────────
    if not settings.SMTP_HOST:
        result["error"] = (
            "SMTP_HOST is empty.  "
            "Open backend/.env and set: SMTP_HOST=smtp.gmail.com"
        )
        result["stages"]["settings_loaded"] = "✗ SMTP_HOST missing"
        return result

    if not settings.SMTP_USER:
        result["error"] = (
            "SMTP_USER is empty.  "
            "Open backend/.env and set: SMTP_USER=your-address@gmail.com"
        )
        result["stages"]["settings_loaded"] = "✗ SMTP_USER missing"
        return result

    if not settings.SMTP_PASSWORD:
        result["error"] = (
            "SMTP_PASSWORD is empty.  "
            "Open backend/.env and set your 16-character Gmail App Password.  "
            "Get one at: myaccount.google.com → Security → App passwords"
        )
        result["stages"]["settings_loaded"] = "✗ SMTP_PASSWORD missing"
        return result

    result["settings_loaded"] = True
    result["stages"]["settings_loaded"] = (
        f"✓ Settings loaded  (host={settings.SMTP_HOST}  "
        f"port={settings.SMTP_PORT}  user={settings.SMTP_USER})"
    )

    # ── Stage 2: TCP connection ────────────────────────────────────────────
    try:
        sock = socket.create_connection(
            (settings.SMTP_HOST, settings.SMTP_PORT), timeout=10
        )
        sock.close()
        result["stages"]["tcp_connect"] = (
            f"✓ TCP connected to {settings.SMTP_HOST}:{settings.SMTP_PORT}"
        )
    except socket.timeout:
        result["stages"]["tcp_connect"] = "✗ TCP timeout — cannot reach server"
        result["error"] = (
            f"Cannot connect to {settings.SMTP_HOST}:{settings.SMTP_PORT}.  "
            "This usually means a firewall is blocking outbound port 587.  "
            "Try from a different network, or switch to port 465."
        )
        return result
    except OSError as exc:
        result["stages"]["tcp_connect"] = f"✗ TCP error: {exc}"
        result["error"] = (
            f"Network error reaching {settings.SMTP_HOST}:{settings.SMTP_PORT}: {exc}"
        )
        return result

    # ── Stages 3-5: SMTP handshake ────────────────────────────────────────
    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:

            # Stage 3 — EHLO
            code, msg = smtp.ehlo()
            if code != 250:
                result["stages"]["ehlo"] = f"✗ EHLO rejected ({code} {msg})"
                result["error"] = f"Server rejected our EHLO greeting: {code} {msg}"
                return result
            result["stages"]["ehlo"] = f"✓ EHLO accepted  (code {code})"

            # Stage 4 — STARTTLS
            smtp.starttls(context=ctx)
            smtp.ehlo()            # ← REQUIRED second EHLO after TLS upgrade
            result["stages"]["starttls"] = "✓ STARTTLS negotiated"

            # Stage 5 — LOGIN
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            result["stages"]["login"] = (
                f"✓ Login successful  ({settings.SMTP_USER})"
            )

    except smtplib.SMTPAuthenticationError as exc:
        code = exc.smtp_code
        detail = exc.smtp_error.decode(errors="ignore") if isinstance(
            exc.smtp_error, bytes
        ) else str(exc.smtp_error)
        result["stages"]["login"] = f"✗ Authentication failed  ({code}: {detail})"
        result["error"] = (
            f"Gmail rejected the login ({code}).  "
            "Make sure you are using a 16-character App Password — "
            "NOT your Gmail account password.  "
            "Generate one at: myaccount.google.com → Security → App passwords.  "
            "Remove any spaces from the 16-character code when pasting into .env."
        )
        return result

    except smtplib.SMTPNotSupportedError as exc:
        result["stages"]["starttls"] = f"✗ STARTTLS not supported: {exc}"
        result["error"] = (
            "Server does not support STARTTLS on this port.  "
            "Try SMTP_PORT=465 with SSL instead."
        )
        return result

    except smtplib.SMTPException as exc:
        result["stages"]["smtp_error"] = f"✗ SMTP error: {exc}"
        result["error"] = f"SMTP protocol error: {exc}"
        return result

    except ssl.SSLError as exc:
        result["stages"]["starttls"] = f"✗ TLS error: {exc}"
        result["error"] = f"TLS/SSL error: {exc}"
        return result

    result["ready"] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_report_html(dashboard_data: Optional[dict] = None) -> str:
    """
    Build a complete HTML BI report from dashboard data.
    If dashboard_data is not supplied it is fetched automatically.
    """
    if dashboard_data is None:
        dashboard_data = build_dashboard_payload()

    kpis        = dashboard_data.get("kpis", [])
    metrics     = dashboard_data.get("model_metrics", {})
    return_dist = dashboard_data.get("return_distribution", {})
    forecast    = dashboard_data.get("forecast_comparison", [])
    generated   = datetime.utcnow().strftime("%d %B %Y, %H:%M UTC")

    # KPI rows
    kpi_rows = ""
    for k in kpis:
        chg = str(k.get("change") or "—")
        color = "#059669" if str(chg).startswith("+") else (
            "#dc2626" if str(chg).startswith("-") or str(chg).startswith("−") else "#6b7280"
        )
        kpi_rows += (
            f'<tr>'
            f'<td style="padding:9px 14px;border-bottom:1px solid #f3f4f6;color:#374151">{k["label"]}</td>'
            f'<td style="padding:9px 14px;border-bottom:1px solid #f3f4f6;font-weight:700;color:#111827">{k["value"]}</td>'
            f'<td style="padding:9px 14px;border-bottom:1px solid #f3f4f6;color:{color};font-weight:500">{chg}</td>'
            f'</tr>'
        )

    clf_m  = metrics.get("return_model", {})
    fcst_m = metrics.get("traffic_model", {})

    def metric_rows(d: dict) -> str:
        if not d:
            return "<tr><td colspan='2' style='color:#9ca3af;font-size:13px'>Train model first</td></tr>"
        return "".join(
            f'<tr>'
            f'<td style="padding:5px 0;color:#6b7280">{k.replace("_"," ").title()}</td>'
            f'<td style="padding:5px 0;font-weight:600;text-align:right;color:#111827">{v}</td>'
            f'</tr>'
            for k, v in d.items()
            if isinstance(v, (int, float, str))
        )

    fc_rows = ""
    for item in (forecast or [])[-5:]:
        err   = abs(item.get("actual", 0) - item.get("predicted", 0))
        arrow = "▲" if item.get("actual", 0) >= item.get("predicted", 0) else "▼"
        fc_rows += (
            f'<tr>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #f3f4f6">{item.get("date","")}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #f3f4f6;text-align:right">{item.get("actual",0):,}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #f3f4f6;text-align:right">{item.get("predicted",0):,}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #f3f4f6;text-align:right;'
            f'color:{"#dc2626" if err > 200 else "#059669"}">{arrow} {err:,}</td>'
            f'</tr>'
        )
    if not fc_rows:
        fc_rows = '<tr><td colspan="4" style="color:#9ca3af;padding:8px">No forecast data yet</td></tr>'

    rr       = return_dist.get("return_rate", 0)
    will_ret = return_dist.get("will_return", "—")
    wont_ret = return_dist.get("wont_return", "—")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WebMine BI Report — {generated}</title>
</head>
<body style="margin:0;padding:20px;background:#f3f4f6;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif">
<div style="max-width:660px;margin:0 auto">

  <!-- Header -->
  <div style="background:#1e3a8a;border-radius:12px 12px 0 0;padding:28px 32px;color:#fff">
    <table style="width:100%"><tr>
      <td>
        <div style="font-size:22px;font-weight:700">📊 WebMine BI</div>
        <div style="margin-top:4px;opacity:.8;font-size:14px">Business Intelligence Report</div>
      </td>
      <td style="text-align:right;vertical-align:top">
        <div style="font-size:12px;opacity:.7">{generated}</div>
      </td>
    </tr></table>
  </div>

  <!-- Body -->
  <div style="background:#fff;padding:0 32px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb">

    <!-- KPIs -->
    <div style="padding:24px 0;border-bottom:1px solid #f3f4f6">
      <h2 style="margin:0 0 16px;font-size:16px;color:#111827;font-weight:600">Key Performance Indicators</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="background:#f9fafb">
          <th style="padding:9px 14px;text-align:left;color:#6b7280;font-size:12px;border-bottom:1px solid #e5e7eb">METRIC</th>
          <th style="padding:9px 14px;text-align:left;color:#6b7280;font-size:12px;border-bottom:1px solid #e5e7eb">VALUE</th>
          <th style="padding:9px 14px;text-align:left;color:#6b7280;font-size:12px;border-bottom:1px solid #e5e7eb">CHANGE</th>
        </tr></thead>
        <tbody>{kpi_rows}</tbody>
      </table>
    </div>

    <!-- Return Users -->
    <div style="padding:24px 0;border-bottom:1px solid #f3f4f6">
      <h2 style="margin:0 0 16px;font-size:16px;color:#111827;font-weight:600">Return User Prediction</h2>
      <table style="width:100%"><tr>
        <td style="width:48%;vertical-align:top">
          <div style="background:#eff6ff;border-radius:10px;padding:16px 20px;border-left:4px solid #2563eb">
            <div style="font-size:11px;color:#3b82f6;font-weight:600;margin-bottom:6px">WILL RETURN</div>
            <div style="font-size:28px;font-weight:700;color:#1d4ed8">{will_ret}</div>
            <div style="font-size:13px;color:#6b7280;margin-top:4px">{rr:.1f}% of users</div>
          </div>
        </td>
        <td style="width:4%"></td>
        <td style="width:48%;vertical-align:top">
          <div style="background:#f9fafb;border-radius:10px;padding:16px 20px;border-left:4px solid #9ca3af">
            <div style="font-size:11px;color:#9ca3af;font-weight:600;margin-bottom:6px">WON'T RETURN</div>
            <div style="font-size:28px;font-weight:700;color:#374151">{wont_ret}</div>
            <div style="font-size:13px;color:#6b7280;margin-top:4px">{100-rr:.1f}% of users</div>
          </div>
        </td>
      </tr></table>
    </div>

    <!-- Model Metrics -->
    <div style="padding:24px 0;border-bottom:1px solid #f3f4f6">
      <h2 style="margin:0 0 16px;font-size:16px;color:#111827;font-weight:600">Model Performance</h2>
      <table style="width:100%"><tr>
        <td style="width:48%;vertical-align:top">
          <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:8px">🎯 Return-User Classifier</div>
          <table style="width:100%;font-size:13px">{metric_rows(clf_m)}</table>
        </td>
        <td style="width:4%"></td>
        <td style="width:48%;vertical-align:top">
          <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:8px">📈 Traffic Forecaster</div>
          <table style="width:100%;font-size:13px">{metric_rows(fcst_m)}</table>
        </td>
      </tr></table>
    </div>

    <!-- Forecast table -->
    <div style="padding:24px 0">
      <h2 style="margin:0 0 16px;font-size:16px;color:#111827;font-weight:600">Recent Traffic — Actual vs Predicted</h2>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead><tr style="background:#f9fafb">
          <th style="padding:6px 10px;text-align:left;color:#6b7280;font-size:11px;border-bottom:1px solid #e5e7eb">DATE</th>
          <th style="padding:6px 10px;text-align:right;color:#6b7280;font-size:11px;border-bottom:1px solid #e5e7eb">ACTUAL</th>
          <th style="padding:6px 10px;text-align:right;color:#6b7280;font-size:11px;border-bottom:1px solid #e5e7eb">PREDICTED</th>
          <th style="padding:6px 10px;text-align:right;color:#6b7280;font-size:11px;border-bottom:1px solid #e5e7eb">ERROR</th>
        </tr></thead>
        <tbody>{fc_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <div style="background:#f9fafb;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 12px 12px;padding:18px 32px;text-align:center">
    <p style="margin:0;font-size:12px;color:#9ca3af;line-height:1.6">
      <strong style="color:#6b7280">WebMine BI</strong> —
      User Web Access Records Mining for Business Intelligence<br>
      Automatically generated — do not reply to this email.<br>
      FYP Project · {datetime.utcnow().year}
    </p>
  </div>

</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# SMTP Email Sender — fixed version
# ─────────────────────────────────────────────────────────────────────────────
def send_report_email(
    to_email: str,
    subject:  str = "WebMine BI — Business Intelligence Report",
    cc_email: Optional[str] = None,
) -> dict:
    """
    Send the HTML BI report to *to_email* via SMTP.

    Parameters
    ----------
    to_email : str
        Who RECEIVES the email.  This is the logged-in user's registered
        address, NOT the SMTP_USER.  It is dynamic and comes from the DB.
    subject : str
        Email subject line.
    cc_email : str | None
        Optional CC address.

    How it works
    ------------
    SMTP_USER  (sender)  ──SMTP──► Gmail servers ──► to_email  (recipient)
    (fixed in .env)                                   (from user DB)

    Raises
    ------
    ValueError  SMTP not configured — .env is incomplete.
    Exception   Detailed SMTP error with exact failure stage and code.
    """
    # ── 1. Run diagnostic first — surface exact failure immediately ────────────
    diag = test_smtp_connection()
    if not diag["ready"]:
        raise Exception(diag.get("error", "SMTP connection test failed."))

    # ── 2. Build MIME message ──────────────────────────────────────────────────
    html_body  = generate_report_html()
    plain_text = (
        f"WebMine BI Report  —  Generated {datetime.utcnow().strftime('%d %B %Y %H:%M UTC')}\n\n"
        "This message contains your BI dashboard report.\n"
        "Please open it in an HTML-capable email client (Gmail, Outlook, Apple Mail).\n"
    )
    sent_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    msg              = MIMEMultipart("alternative")
    msg["Subject"]   = subject
    msg["From"]      = f"{settings.EMAIL_FROM} <{settings.SMTP_USER}>"
    msg["To"]        = to_email
    msg["Reply-To"]  = settings.SMTP_USER
    if cc_email:
        msg["Cc"]    = cc_email

    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_body,  "html",  "utf-8"))

    recipients = [to_email] + ([cc_email] if cc_email else [])

    # ── 3. Send — with correct EHLO sequence ──────────────────────────────────
    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as smtp:
            smtp.ehlo()                 # greeting before STARTTLS
            smtp.starttls(context=ctx)  # upgrade to TLS
            smtp.ehlo()                 # ← REQUIRED second greeting after TLS
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            smtp.send_message(msg)      # correct method for MIMEMultipart

    except smtplib.SMTPAuthenticationError as exc:
        code   = exc.smtp_code
        detail = (
            exc.smtp_error.decode(errors="ignore")
            if isinstance(exc.smtp_error, bytes)
            else str(exc.smtp_error)
        )
        raise Exception(
            f"Gmail rejected login ({code}: {detail}).  "
            "Use an App Password, not your Gmail account password.  "
            "Get one at: myaccount.google.com → Security → App passwords"
        )

    except smtplib.SMTPRecipientsRefused as exc:
        refused = {k: v for k, v in exc.recipients.items()}
        raise Exception(
            f"Recipient address refused by Gmail: {refused}.  "
            "Check that the To address is a valid email."
        )

    except smtplib.SMTPSenderRefused as exc:
        raise Exception(
            f"Sender address refused ({exc.smtp_code}: {exc.smtp_error}).  "
            f"Verify SMTP_USER={settings.SMTP_USER} is correct in .env."
        )

    except smtplib.SMTPDataError as exc:
        raise Exception(
            f"Gmail rejected the message body ({exc.smtp_code}: {exc.smtp_error})."
        )

    except smtplib.SMTPException as exc:
        raise Exception(f"SMTP protocol error: {exc}")

    except ssl.SSLError as exc:
        raise Exception(f"TLS/SSL error while connecting: {exc}")

    log.info("Report sent  from=%s  to=%s  cc=%s", settings.SMTP_USER, to_email, cc_email)
    return {
        "status":  "sent",
        "to":      to_email,
        "subject": subject,
        "sent_at": sent_at,
        "message": f"Report successfully sent to {to_email}",
    }
