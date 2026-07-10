"""backend/tests/test_predictions.py — predictions endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_return_requires_model():
    payload = {"users": [{"session_duration": 5.0, "page_views": 8,
                           "visit_frequency": 10, "bounce_rate": 0.3,
                           "navigation_depth": 2.0}]}
    resp = client.post("/api/v1/predict-return", json=payload)
    # Either 200 (model loaded) or 409 (model not trained yet)
    assert resp.status_code in (200, 409)

def test_predict_return_invalid_bounce_rate():
    payload = {"users": [{"session_duration": 5.0, "page_views": 8,
                           "visit_frequency": 10, "bounce_rate": 1.5,
                           "navigation_depth": 2.0}]}
    resp = client.post("/api/v1/predict-return", json=payload)
    # bounce_rate is clamped by validator, should not be a 422
    assert resp.status_code in (200, 409)
