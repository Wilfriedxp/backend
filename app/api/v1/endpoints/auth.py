"""
backend/app/api/v1/endpoints/auth.py
Auth endpoints: register, login, me, update-profile,
change-password, regenerate-collector-token.
"""
from __future__ import annotations
import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.app_user import AppUser
from app.schemas.auth import (
    ChangePasswordRequest, LoginRequest, RegisterRequest,
    TokenResponse, UpdateProfileRequest, UserOut,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
log    = logging.getLogger("endpoint.auth")
_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Dependency ────────────────────────────────────────────────────────────────
def get_current_user(
    token: str = Depends(_oauth2),
    db:    Session = Depends(get_db),
) -> AppUser:
    email = decode_access_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(AppUser).filter(AppUser.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled.")
    return user


# ── Register ──────────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.query(AppUser).filter(AppUser.email == str(body.email).lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered.")
    user = AppUser(
        full_name       = body.full_name.strip(),
        email           = str(body.email).lower(),
        hashed_password = hash_password(body.password),
        collector_token = secrets.token_urlsafe(32),
    )
    db.add(user); db.commit(); db.refresh(user)
    log.info("Registered: %s", user.email)
    return TokenResponse(
        access_token=create_access_token(user.email),
        user=UserOut.model_validate(user),
    )


# ── Login ─────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(AppUser).filter(AppUser.email == str(body.email).lower()).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled.")
    log.info("Login: %s", user.email)
    return TokenResponse(
        access_token=create_access_token(user.email),
        user=UserOut.model_validate(user),
    )


# ── Me ────────────────────────────────────────────────────────────────────────
@router.get("/me", response_model=UserOut)
def me(current_user: AppUser = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)


# ── Update profile ────────────────────────────────────────────────────────────
@router.put("/profile", response_model=UserOut)
def update_profile(
    body:         UpdateProfileRequest,
    current_user: AppUser  = Depends(get_current_user),
    db:           Session  = Depends(get_db),
) -> UserOut:
    """Update profile fields and/or toggle auto_email_reports."""
    if body.full_name          is not None: current_user.full_name          = body.full_name.strip()
    if body.phone              is not None: current_user.phone              = body.phone.strip() or None
    if body.company            is not None: current_user.company            = body.company.strip() or None
    if body.website            is not None: current_user.website            = body.website.strip() or None
    if body.bio                is not None: current_user.bio                = body.bio.strip() or None
    if body.auto_email_reports is not None: current_user.auto_email_reports = body.auto_email_reports
    db.commit(); db.refresh(current_user)
    log.info("Profile updated: %s  auto_email=%s", current_user.email, current_user.auto_email_reports)
    return UserOut.model_validate(current_user)


# ── Change password ───────────────────────────────────────────────────────────
@router.put("/change-password")
def change_password(
    body:         ChangePasswordRequest,
    current_user: AppUser  = Depends(get_current_user),
    db:           Session  = Depends(get_db),
) -> dict:
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()
    return {"message": "Password changed successfully."}


# ── Regenerate collector token ────────────────────────────────────────────────
@router.post("/regenerate-collector-token", response_model=UserOut)
def regenerate_collector_token(
    current_user: AppUser  = Depends(get_current_user),
    db:           Session  = Depends(get_db),
) -> UserOut:
    """Generate a new collector token for the Chrome extension.
    Old token is immediately invalidated."""
    current_user.collector_token = secrets.token_urlsafe(32)
    db.commit(); db.refresh(current_user)
    log.info("Collector token regenerated: %s", current_user.email)
    return UserOut.model_validate(current_user)
