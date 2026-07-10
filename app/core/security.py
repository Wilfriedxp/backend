"""
backend/app/core/security.py
Password hashing (bcrypt direct) and JWT token creation / verification.
Uses bcrypt directly to avoid the passlib + bcrypt>=4.0 compatibility issue.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from app.core.config import settings

log = logging.getLogger("security")

# ── Password hashing ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

# ── JWT tokens ────────────────────────────────────────────────────────────────
ALGORITHM                = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

def create_access_token(subject: str) -> str:
    expire  = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": subject, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
