"""Centralised application configuration.

All configuration is sourced from environment variables (optionally loaded from a
`.env` file). Nothing machine-specific is hardcoded: paths are resolved relative
to the repository root so the project is portable across machines.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/cybersentinel/utils/config.py -> repo root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings for every CyberSentinel component."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # --- LLM ---
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_adapter_path: str | None = None
    llm_backend: Literal["mock", "hf"] = "mock"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.2
    llm_load_in_4bit: bool = True
    llm_device: str = "auto"
    hf_token: str | None = None

    # --- RAG ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "cybersentinel_kb"
    qdrant_in_memory: bool = False
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_backend: Literal["hash", "sentence-transformers"] = "hash"
    top_k: int = 5
    # Hash embeddings produce lower absolute cosine scores than a trained
    # encoder, so the default floor is deliberately permissive. Raise it when
    # switching EMBEDDING_BACKEND to sentence-transformers.
    rag_score_threshold: float = 0.05

    # --- Database ---
    database_url: str = "sqlite:///./cybersentinel.db"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    max_input_chars: int = 20_000
    max_upload_bytes: int = 1_048_576

    # --- Human-in-the-loop ---
    approval_severity_threshold: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "HIGH"

    # --- Derived paths (not env-driven, but overridable) ---
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @field_validator("model_adapter_path", "hf_token", "qdrant_api_key", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # --- Convenience path helpers ---
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.data_dir / "knowledge_base"

    @property
    def evaluation_dir(self) -> Path:
        return self.project_root / "evaluation"

    def resolve(self, path: str | Path) -> Path:
        """Resolve a possibly-relative path against the project root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings. Call `get_settings.cache_clear()` in tests."""
    return Settings()
