"""
backend/app/db/init_db.py
Creates all database tables on first run.
Call init_db() once from the FastAPI lifespan hook.
"""
from app.db.session import engine
from app.db.base import Base


def init_db() -> None:
    """Create all tables defined in ORM models (idempotent)."""
    Base.metadata.create_all(bind=engine)
