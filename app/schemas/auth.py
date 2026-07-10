"""
backend/app/schemas/auth.py
Pydantic v2 schemas for authentication and user profile.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    full_name: str      = Field(..., min_length=2, max_length=100)
    email:     EmailStr
    password:  str      = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_not_all_digits(cls, v: str) -> str:
        if v.isdigit():
            raise ValueError("Password must not be all digits.")
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    id:                 int
    full_name:          str
    email:              str
    is_active:          bool
    phone:              Optional[str] = None
    company:            Optional[str] = None
    website:            Optional[str] = None
    bio:                Optional[str] = None
    auto_email_reports: bool          = False
    collector_token:    str
    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut


class UpdateProfileRequest(BaseModel):
    full_name:          Optional[str]  = Field(None, min_length=2, max_length=100)
    phone:              Optional[str]  = Field(None, max_length=30)
    company:            Optional[str]  = Field(None, max_length=150)
    website:            Optional[str]  = Field(None, max_length=255)
    bio:                Optional[str]  = Field(None, max_length=500)
    auto_email_reports: Optional[bool] = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=8, max_length=128)
