"""
backend/app/api/v1/router.py
"""
from fastapi import APIRouter
from app.api.v1.endpoints import auth, upload, predictions, dashboard, reports, collect

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(upload.router,      tags=["Data Ingestion"])
api_router.include_router(predictions.router, tags=["ML Models"])
api_router.include_router(dashboard.router,   tags=["Dashboard"])
api_router.include_router(reports.router)
api_router.include_router(collect.router)
