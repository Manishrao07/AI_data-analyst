"""
Centralized configuration for the AI Data Analyst backend.
All secrets are read from environment variables -- nothing is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM configuration (Groq -- fast + free tier, OpenAI-compatible API)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # App
    APP_NAME: str = "AI Data Analyst"
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    MAX_ROWS_PREVIEW: int = 20

    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./storage/uploads")
    SESSION_DIR: str = os.getenv("SESSION_DIR", "./storage/sessions")

    # Cache
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.SESSION_DIR, exist_ok=True)
