"""
backend/app/core/config.py
All application settings loaded from backend/.env

THE .env FILE MUST BE AT:  webmine-project/backend/.env
The path is resolved relative to this config.py file, so it works
regardless of which directory you run uvicorn from.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always points to  backend/.env  no matter where uvicorn is launched from
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),          # absolute path — never breaks
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",                    # silently ignore unknown keys
    )

    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME:    str  = "WebMine BI API"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False

    # ── CORS ────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://webmine-five.vercel.app/login" ,
    ]

    # ── Database ─────────────────────────────────────────────────────────────
    # SQLite default (zero setup). Switch to MySQL in .env:
    # DATABASE_URL=mysql+pymysql://user:password@localhost:3306/webmine
    DATABASE_URL: str = "sqlite:///./webmine.db"

    # ── ML model paths ───────────────────────────────────────────────────────
    ML_DIR: Path = Path(__file__).resolve().parents[1] / "ml"

    @property
    def return_model_path(self) -> Path:
        return self.ML_DIR / "models" / "return_user_classifier.pkl"

    @property
    def traffic_model_path(self) -> Path:
        return self.ML_DIR / "models" / "traffic_forecast_model.pkl"

    # ── Auth / JWT ───────────────────────────────────────────────────────────
    # Generate once:  python -c "import secrets; print(secrets.token_hex(32))"
    # Paste the output into backend/.env as SECRET_KEY=<value>
    SECRET_KEY: str = "change-me-generate-with-secrets-token-hex-32"

    # ── Upload ───────────────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 50

    # ── SMTP Email ───────────────────────────────────────────────────────────
    # These MUST be set in backend/.env — defaults are empty on purpose
    # so the diagnostic endpoint can tell you what is missing.
    #
    # Gmail setup:
    #   SMTP_HOST     = smtp.gmail.com
    #   SMTP_PORT     = 587
    #   SMTP_USER     = your-address@gmail.com   ← who SENDS the email
    #   SMTP_PASSWORD = 16-char-app-password     ← NOT your Gmail login password
    #   EMAIL_FROM    = WebMine BI               ← display name in From field
    #
    # The RECIPIENT (to_email) is the logged-in user's email from the database,
    # NOT from this file.
    SMTP_HOST:     str = ""
    SMTP_PORT:     int = 587
    SMTP_USER:     str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM:    str = "WebMine BI"

    def env_file_location(self) -> str:
        """Return the absolute path where .env is expected — useful for diagnostics."""
        return str(_ENV_FILE)


settings = Settings()
