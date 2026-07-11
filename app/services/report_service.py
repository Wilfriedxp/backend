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
import resend
from datetime import datetime
from typing import Dict, Optional

from app.core.config import settings
from app.services.dashboard_service import build_dashboard_payload

log = logging.getLogger("report_service")


# ─────────────────────────────────────────────────────────────────────────────
# SMTP Diagnostic — call this first to find exactly what is broken
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("report_service")
resend.api_key = settings.RESEND_API_KEY




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
    subject: str = "WebMine BI — Business Intelligence Report",
    cc_email: Optional[str] = None,
) -> dict:
    """
    Send the BI report using Resend.

    Parameters
    ----------
    to_email
        Recipient email.

    subject
        Email subject.

    cc_email
        Optional CC recipient.
    """

    html_body = generate_report_html()

    params = {
        "from": settings.EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    }

    if cc_email:
        params["cc"] = [cc_email]

    try:
        response = resend.Emails.send(params)

        log.info(
            "Report sent successfully from=%s to=%s",
            settings.EMAIL_FROM,
            to_email,
        )

        return {
            "status": "sent",
            "to": to_email,
            "subject": subject,
            "message": f"Report successfully sent to {to_email}",
            "provider": "Resend",
            "response": response,
        }

    except Exception as exc:
        log.exception("Failed to send email")

        raise Exception(
            f"Unable to send email using Resend: {exc}"
        )
