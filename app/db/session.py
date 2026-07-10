"""
backend/app/db/session.py
SQLAlchemy engine and session factory.
Uses SQLite by default; set DATABASE_URL in .env to switch to MySQL:
    DATABASE_URL=mysql+pymysql://user:password@localhost:3306/webmine
"""
from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency.
    Creates a database session and closes it automatically.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
