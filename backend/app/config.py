from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(min_length=1)
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    QDRANT_URL: str = Field(min_length=1)
    QDRANT_API_KEY: str = Field(min_length=1)
    QDRANT_COLLECTION: str = "axiom-passages"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVAL_FALLBACK_THRESHOLD: float = Field(default=0.35, ge=0, le=1)
    DATABASE_URL: str = Field(min_length=1)
    TAVILY_API_KEY: str = Field(min_length=1)

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
