"""
backend/app/db/base.py
Declarative base shared by all ORM models.
Import every model here so Alembic can auto-detect them.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import all models so Base.metadata knows about every table
from app.models import access_log, prediction, report, user   # noqa: F401, E402
from app.models import app_user                                # noqa: F401, E402
