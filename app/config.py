import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

IS_VERCEL = os.environ.get("VERCEL") == "1" or not os.access(Path(__file__).resolve().parent.parent, os.W_OK)
DEFAULT_DB_URL = "sqlite+aiosqlite:////tmp/sehtak_ai.db" if IS_VERCEL else "sqlite+aiosqlite:///./sehtak_ai.db"
DEFAULT_REPORTS_DIR = Path("/tmp/generated_reports") if IS_VERCEL else (Path(__file__).resolve().parent.parent / "generated_reports")


class Settings(BaseSettings):
    """Centralized configuration for Sehtak AI (Local, Docker, and Vercel Serverless ready)."""

    APP_NAME: str = "Sehtak AI - Proactive Medical Safety & Triage Guardian"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = DEFAULT_DB_URL

    # AI Providers
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = ""
    ENABLE_TELEGRAM_POLLING: bool = False

    # WhatsApp Cloud API
    WHATSAPP_VERIFY_TOKEN: str = "sehtak-whatsapp-verify-secret-2026"
    WHATSAPP_ACCESS_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    REPORTS_OUTPUT_DIR: Path = DEFAULT_REPORTS_DIR

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
if IS_VERCEL and settings.DATABASE_URL.startswith("sqlite+aiosqlite:///./"):
    settings.DATABASE_URL = "sqlite+aiosqlite:////tmp/sehtak_ai.db"
    settings.REPORTS_OUTPUT_DIR = Path("/tmp/generated_reports")

os.makedirs(settings.REPORTS_OUTPUT_DIR, exist_ok=True)
