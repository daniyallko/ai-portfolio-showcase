import re
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default="mock-key", description="Google Gemini API Key")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai",
        description="Async PostgreSQL connection URL"
    )
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIMENSIONS: int = 768
    CHAT_MODEL: str = "gemini-2.5-flash"
    RRF_K: int = 60
    TOP_K_CHUNKS: int = 4
    PORT: int = 8000
    DEBUG: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        if isinstance(v, str):
            # Normalize driver
            if v.startswith("postgres://"):
                v = "postgresql+asyncpg://" + v[len("postgres://"):]
            elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
                v = "postgresql+asyncpg://" + v[len("postgresql://"):]
            # Normalize sslmode for asyncpg
            v = re.sub(r"[?&]sslmode=([^&]+)", r"?ssl=\1", v)
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
