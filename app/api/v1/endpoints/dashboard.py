"""
dashboard.py  —  backend/app/api/v1/endpoints/dashboard.py
Single GET endpoint that returns all KPIs and chart data for the React dashboard.
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, status
from app.services.dashboard_service import build_dashboard_payload

router = APIRouter()
log    = logging.getLogger("endpoint.dashboard")


@router.get(
    "/dashboard-data",
    summary="Fetch all dashboard KPIs and chart data",
    description=(
        "Returns KPI cards, traffic trend, return-user distribution, "
        "actual-vs-predicted comparison, and model performance metrics. "
        "Renders synthetic demo data when no CSV has been uploaded yet."
    ),
)
async def get_dashboard_data() -> dict:
    try:
        return build_dashboard_payload()
    except Exception as exc:
        log.exception("Dashboard assembly failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard error: {exc}",
        )
