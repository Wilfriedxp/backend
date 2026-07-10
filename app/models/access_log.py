"""
backend/app/models/access_log.py  — raw web access log record
backend/app/models/user.py        — hashed user identity
backend/app/models/prediction.py  — ML prediction result
backend/app/models/report.py      — generated report metadata

All four models live in one file for conciseness; split them into
separate files matching the project structure if preferred.
"""
# ── access_log.py content ──────────────────────────────────────────────────────
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func
from app.db.base import Base   # noqa: imported by base.py


class AccessLog(Base):
    __tablename__ = "access_logs"
    id          = Column(Integer, primary_key=True, index=True)
    ip_address  = Column(String(45),  nullable=False, index=True)
    user_id     = Column(String(64),  nullable=True,  index=True)  # hashed/anon
    timestamp   = Column(DateTime,    nullable=False, index=True)
    url         = Column(String(2048),nullable=False)
    method      = Column(String(10),  nullable=True)
    status_code = Column(Integer,     nullable=True)
    user_agent  = Column(Text,        nullable=True)
    referrer    = Column(String(2048),nullable=True)
    bytes_sent  = Column(Integer,     nullable=True)
    created_at  = Column(DateTime,    server_default=func.now())

    __table_args__ = (Index("ix_al_ip_ts", "ip_address", "timestamp"),)


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(String(64), unique=True, nullable=False, index=True)
    first_seen      = Column(DateTime, nullable=True)
    last_seen       = Column(DateTime, nullable=True)
    session_count   = Column(Integer, default=0)
    page_view_total = Column(Integer, default=0)
    created_at      = Column(DateTime, server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(String(64), nullable=False, index=True)
    prediction_type = Column(String(32), nullable=False)  # "return" | "traffic"
    predicted_value = Column(Float,      nullable=False)
    probability     = Column(Float,      nullable=True)
    model_version   = Column(String(16), nullable=True)
    created_at      = Column(DateTime,   server_default=func.now())


class Report(Base):
    __tablename__ = "reports"
    id           = Column(Integer,     primary_key=True, index=True)
    report_type  = Column(String(32),  nullable=False, default="full")
    sent_to      = Column(String(255), nullable=True)
    generated_at = Column(DateTime,    nullable=False, default=datetime.utcnow)
    file_path    = Column(String(512), nullable=True)
    filters_json = Column(Text,        nullable=True)
    status       = Column(String(16),  nullable=False, default="generated")
