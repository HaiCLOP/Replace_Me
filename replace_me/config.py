"""Settings from .env (see .env.example for what each one does) and the paths of your data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    # models
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    LLM_CHAIN: str = "anthropic:claude-opus-5-5@medium,anthropic:claude-sonnet-5-5"
    RESEARCH_MODEL: str = "anthropic:claude-opus-5-5@medium"
    PICKER_CHAIN: str = "anthropic:claude-opus-5-5@low"
    JUDGE_CHAIN: str = "anthropic:claude-opus-5-5@low"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_TIMEOUT_SECONDS: float = 60.0
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # reply pipeline (defaults = best measured setup, docs/EVALUATION.md)
    CANDIDATES: int = Field(default=4, ge=1, le=6)
    PICKER: str = "judge"
    USE_TOOLS: bool = True
    MASK_NAMES: bool = True
    GOSSIP_MIN_SCORE: int = 90
    VOICE: str = ""
    VOICE_MODEL: str = "ollama:my-voice"

    # instagram
    INBOX_SCAN_SECONDS: float = 5.0
    INBOX_MAX_PARALLEL: int = 3
    INBOX_HOT_SECONDS: float = 120.0
    INBOX_HOT_POLL_SECONDS: float = 5.0
    BURST_SETTLE_SECONDS: float = 3.0
    HUMAN_TIMING: bool = True
    IGNORE_HANDLES: str = ""
    NAVIGATION_TIMEOUT_MS: int = 45000

    # paths (relative to the repo)
    ME_DIR: Path = ROOT / "me"
    DATA_DIR: Path = ROOT / ".data"
    BROWSER_PROFILE: Path = ROOT / "browser_profile"

    @property
    def export_dir(self) -> Path:
        return self.ME_DIR / "export"

    @property
    def people_dir(self) -> Path:
        return self.ME_DIR / "people"

    @property
    def db_path(self) -> Path:
        return self.DATA_DIR / "live.db"

    def ignored(self) -> set[str]:
        return {h.strip().lower().lstrip("@") for h in self.IGNORE_HANDLES.split(",") if h.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return s
