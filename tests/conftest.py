from __future__ import annotations

import pytest

from mia import runtime
from mia.config import Settings
from mia.llm import LLM, RawResponse


class FakeBackend:
    """Stands in for a provider. Returns queued responses and records every call."""

    def __init__(self, responses: list[str | Exception] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def generate(self, model, prompt, system, schema, temperature) -> RawResponse:
        self.calls.append({"model": model, "prompt": prompt, "system": system, "schema": schema})
        item = self.responses.pop(0) if self.responses else "ok"
        if isinstance(item, Exception):
            raise item
        return RawResponse(text=item, input_tokens=10, output_tokens=5)

    def is_retryable(self, exc: Exception) -> bool:
        return isinstance(exc, TransientError)


class TransientError(Exception):
    pass


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        model_bulk="fake/bulk-model",
        model_smart="fake/smart-model",
        gemini_api_key="",
        tavily_api_key="",
        llm_rpm=60_000,
        max_llm_calls_per_run=3,
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def llm(settings, backend) -> LLM:
    return LLM(settings, backends={"fake": backend}, sleep=lambda s: None)


@pytest.fixture(autouse=True)
def _reset_runtime():
    yield
    runtime._settings, runtime._offline, runtime._llm = None, False, None
