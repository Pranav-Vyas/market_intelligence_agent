"""Per-run context shared by all modules: settings, offline mode and the LLM client.

The pipeline calls `configure()` once at the start of a run. Modules then read from here
instead of passing these through every function signature.
"""

from __future__ import annotations

from mia.config import Settings, get_settings
from mia.llm import LLM

_settings: Settings | None = None
_offline: bool = False
_llm: LLM | None = None


def configure(
    *, settings: Settings | None = None, offline: bool = False, llm_client: LLM | None = None
) -> None:
    global _settings, _offline, _llm
    _settings = settings or get_settings()
    _offline = offline
    _llm = llm_client or LLM(_settings, offline=offline)


def settings() -> Settings:
    return _settings or get_settings()


def is_offline() -> bool:
    return _offline


def llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM(settings(), offline=_offline)
    return _llm
