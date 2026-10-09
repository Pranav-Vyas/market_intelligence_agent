"""Settings: model names, limits, thresholds, score weights and paths.

Secrets and per-person values come from `.env` (see `.env.example`). Everything else has a
default here. Tests build `Settings(...)` directly instead of reading the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # LLM: model strings are "<provider>/<model-id>", e.g. "gemini/<id from `mia models`>".
    model_bulk: str = field(default_factory=lambda: _env("MIA_MODEL_BULK"))
    model_smart: str = field(default_factory=lambda: _env("MIA_MODEL_SMART"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    llm_rpm: int = field(default_factory=lambda: _env_int("MIA_LLM_RPM", 10))
    max_llm_calls_per_run: int = field(
        default_factory=lambda: _env_int("MIA_MAX_LLM_CALLS_PER_RUN", 200)
    )

    data_dir: Path = field(default_factory=lambda: Path(_env("MIA_DATA_DIR") or REPO_ROOT / "data"))

    # Agent and collection limits (SPEC.md §3, §5)
    min_competitors: int = 5
    max_competitors: int = 8
    max_agent_steps: int = 20
    min_complaint_docs_per_company: int = 15
    max_complaint_docs_per_source: int = 40

    # Evidence gates and scoring (SPEC.md §4)
    max_complaints_per_doc: int = 3
    gate_min_complaints: int = 5
    gate_min_docs: int = 3
    gate_min_source_types: int = 2
    unmet_need_coverage_below: float = 0.4
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"F": 0.35, "S": 0.25, "G": 0.25, "E": 0.15}
    )

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


@cache
def get_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    return Settings()
