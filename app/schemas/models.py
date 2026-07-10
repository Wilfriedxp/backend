"""
backend/app/schemas/models.py
Pydantic v2 request / response schemas for every API endpoint.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    message:         str
    rows_ingested:   int
    users_found:     int
    date_range:      Dict[str, str]
    feature_preview: List[Dict[str, Any]]
    upload_stats:    Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Return-user model
# ─────────────────────────────────────────────────────────────────────────────
class TrainReturnModelResponse(BaseModel):
    message:              str
    users_trained_on:     int
    test_metrics:         Dict[str, Any]
    cv_summary:           Dict[str, float]
    feature_importances:  Dict[str, float]


class UserFeatures(BaseModel):
    session_duration: float = Field(..., ge=0,    description="Mean session length in minutes")
    page_views:       int   = Field(..., ge=1,    description="Total page views")
    visit_frequency:  int   = Field(..., ge=1,    description="Number of distinct sessions")
    bounce_rate:      float = Field(..., ge=0.0, le=1.0, description="Bounce rate [0–1]")
    navigation_depth: float = Field(..., ge=0,    description="Mean URL path depth")

    @field_validator("bounce_rate")
    @classmethod
    def clamp_bounce(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)


class ReturnPredictionRequest(BaseModel):
    users: List[UserFeatures] = Field(..., min_length=1, max_length=500)


class SinglePrediction(BaseModel):
    will_return:     bool
    probability:     float
    label:           str
    confidence_band: str      # HIGH / MEDIUM / LOW


class ReturnPredictionResponse(BaseModel):
    predictions:   List[SinglePrediction]
    model_version: str


# ─────────────────────────────────────────────────────────────────────────────
# Traffic model
# ─────────────────────────────────────────────────────────────────────────────
class TrainTrafficModelResponse(BaseModel):
    message:              str
    training_days:        int
    test_metrics:         Dict[str, float]
    cv_summary:           Dict[str, float]
    feature_importances:  Dict[str, float]


class TrafficPredictionResponse(BaseModel):
    date:               str
    predicted_visitors: int
    confidence_low:     int
    confidence_high:    int
    std_deviation:      float
    model_version:      str


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
class KPICard(BaseModel):
    label:  str
    value:  Any
    change: Optional[float] = None
    unit:   Optional[str]   = None


class TrafficPoint(BaseModel):
    date:      str
    visits:    int
    predicted: Optional[int] = None


class DashboardData(BaseModel):
    kpis:                List[KPICard]
    traffic_trend:        List[TrafficPoint]
    return_distribution:  Dict[str, Any]
    forecast_comparison:  List[Dict[str, Any]]
    model_metrics:        Dict[str, Any]
    data_available:       bool


# ─────────────────────────────────────────────────────────────────────────────
# Reports & Email
# ─────────────────────────────────────────────────────────────────────────────
class SendReportEmailRequest(BaseModel):
    # to_email is OPTIONAL — if omitted, the endpoint uses the logged-in user's
    # registered email address automatically.
    to_email: Optional[EmailStr] = Field(
        None,
        description=(
            "Recipient address. Leave empty to send to your own registered email. "
            "Supply a different address to forward the report to a supervisor."
        ),
    )
    subject: str = Field(
        default="WebMine BI — Business Intelligence Report",
        description="Email subject line",
    )
    cc_email: Optional[EmailStr] = Field(
        None,
        description="Optional CC address",
    )


class SendReportEmailResponse(BaseModel):
    status:   str   # "sent" | "failed"
    to:       str
    subject:  str
    sent_at:  str
    message:  str


class GenerateReportResponse(BaseModel):
    message:       str
    report_html:   str          # full HTML string of the report
    generated_at:  str
    kpi_count:     int


# ─────────────────────────────────────────────────────────────────────────────
# Generic
# ─────────────────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str


class StatusResponse(BaseModel):
    status:  str
    message: str
