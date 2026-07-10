"""backend/tests/test_upload.py — upload endpoint tests."""
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_CSV = (
    "ip_address,timestamp,method,url,status_code,bytes_sent,user_agent\n"
    "10.0.0.1,2024-01-15 10:00:00,GET,/home,200,1200,Mozilla/5.0\n"
    "10.0.0.2,2024-01-15 10:01:00,GET,/about,200,900,Mozilla/5.0\n"
    "10.0.0.1,2024-01-15 10:05:00,GET,/products,200,1500,Mozilla/5.0\n"
)

def test_upload_valid_csv():
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("test.csv", io.BytesIO(SAMPLE_CSV.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows_ingested"] > 0
    assert "users_found" in data

def test_upload_non_csv_rejected():
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 415

def test_upload_missing_columns():
    bad_csv = b"col1,col2\nval1,val2\n"
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
    )
    assert resp.status_code == 422
