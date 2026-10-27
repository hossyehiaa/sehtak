import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is explicitly loaded from project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)

IS_VERCEL = os.environ.get("VERCEL") == "1" or not os.access(Path(__file__).resolve().parent.parent, os.W_OK)
DEFAULT_DB_URL = "sqlite+aiosqlite:////tmp/sehtak_ai.db" if IS_VERCEL else "sqlite+aiosqlite:///./sehtak_ai.db"
DEFAULT_REPORTS_DIR = Path("/tmp/generated_reports") if IS_VERCEL else (Path(__file__).resolve().parent.parent / "generated_reports")
DEFAULT_CHROMA_DIR = "/tmp/chroma_db" if IS_VERCEL else "./data/vector_db"


class Settings(BaseSettings):
    """Centralized configuration for Sehtak AI v2."""

    APP_NAME: str = "Sehtak AI - Proactive Medical Safety & Triage Guardian"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = DEFAULT_DB_URL

    # Auth & JWT Security (Task 1)
    JWT_SECRET: str = "sehtak-super-secret-jwt-key-2026-production-ready"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MIN: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALLOW_ANONYMOUS_DEMO: bool = True

    # AI Providers
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # RAG Grounding Configuration (Task 2)
    CHROMA_PERSIST_DIR: str = DEFAULT_CHROMA_DIR
    RAG_MIN_CONFIDENCE: float = 0.76
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
    settings.CHROMA_PERSIST_DIR = "/tmp/chroma_db"

os.makedirs(settings.REPORTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
