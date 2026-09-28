import pytest
from core.config import Settings

def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai")

    settings = Settings()
    assert settings.GEMINI_API_KEY == "test-key-12345"
    assert settings.DATABASE_URL == "postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai"
    assert settings.EMBEDDING_MODEL == "models/text-embedding-004"
    assert settings.EMBEDDING_DIMENSIONS == 768
    assert settings.CHAT_MODEL == "gemini-2.5-flash"

def test_settings_normalizes_postgres_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@ep-cool.neon.tech/neondb?sslmode=require")
    settings = Settings()
    assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
    assert "ssl=require" in settings.DATABASE_URL

