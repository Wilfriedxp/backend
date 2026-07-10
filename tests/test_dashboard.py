"""backend/tests/test_dashboard.py — dashboard endpoint tests."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_dashboard_returns_demo_data():
    resp = client.get("/api/v1/dashboard-data")
    assert resp.status_code == 200
    data = resp.json()
    assert "kpis" in data
    assert "traffic_trend" in data
    assert "return_distribution" in data
    assert isinstance(data["kpis"], list)
    assert len(data["kpis"]) > 0

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
