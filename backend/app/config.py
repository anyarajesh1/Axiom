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
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ENTAILMENT_MODEL: str = "cross-encoder/nli-MiniLM2-L6-H768"
    RETRIEVAL_FALLBACK_THRESHOLD: float = Field(default=0.5, ge=0, le=1)
    USE_HF_INFERENCE_API: bool = False
    HF_TOKEN: str | None = None
    DATABASE_URL: str = Field(min_length=1)
    TAVILY_API_KEY: str = Field(min_length=1)
    FRONTEND_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def frontend_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.FRONTEND_ORIGINS.split(",")
            if origin.strip()
        ]

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
