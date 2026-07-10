"""
backend/app/models/app_user.py
SQLAlchemy model for BI dashboard users.
"""
from __future__ import annotations
import secrets
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func
from app.db.base import Base


class AppUser(Base):
    __tablename__ = "app_users"

    id                  = Column(Integer,     primary_key=True, index=True)
    # Core identity
    full_name           = Column(String(100), nullable=False)
    email               = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password     = Column(String(255), nullable=False)
    is_active           = Column(Boolean,     default=True, nullable=False)
    # Extended profile
    phone               = Column(String(30),  nullable=True)
    company             = Column(String(150), nullable=True)
    website             = Column(String(255), nullable=True)
    bio                 = Column(Text,        nullable=True)
    # Settings
    auto_email_reports  = Column(Boolean, default=False, nullable=False)
    # Chrome extension data-collection token (unique per user, never a JWT)
    collector_token     = Column(String(64), unique=True, nullable=False,
                                 default=lambda: secrets.token_urlsafe(32))
    # Timestamps
    created_at          = Column(DateTime, server_default=func.now())
    updated_at          = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<AppUser id={self.id} email={self.email!r}>"
