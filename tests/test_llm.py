from __future__ import annotations

import pytest
from pydantic import BaseModel

from mia.llm import (
    LLM,
    LLMBudgetExceeded,
    LLMCacheMiss,
    LLMConfigError,
    LLMOutputError,
    LLMUnavailable,
)
from tests.conftest import FakeBackend, TransientError


class Item(BaseModel):
    name: str
    severity: int


def test_complete_resolves_alias_and_counts_usage(llm, backend):
    backend.responses = ["hello"]
    assert llm.complete("hi", model="smart") == "hello"
    assert backend.calls[0]["model"] == "smart-model"
    assert llm.usage.requests == 1
    assert llm.usage.input_tokens == 10
    assert llm.usage.by_model == {"fake/smart-model": 1}


def test_second_identical_call_hits_cache(llm, backend):
    backend.responses = ["first", "second"]
    assert llm.complete("same prompt") == "first"
    assert llm.complete("same prompt") == "first"
    assert len(backend.calls) == 1
    assert llm.usage.cache_hits == 1


def test_cache_is_shared_across_instances(settings, llm, backend):
    backend.responses = ["cached"]
    llm.complete("prompt")
    offline = LLM(settings, offline=True, backends={"fake": FakeBackend()})
    assert offline.complete("prompt") == "cached"


def test_offline_cache_miss_raises(settings):
    offline = LLM(settings, offline=True, backends={"fake": FakeBackend()})
    with pytest.raises(LLMCacheMiss):
        offline.complete("never seen")


def test_budget_is_enforced(llm, backend):
    for i in range(3):
        llm.complete(f"prompt {i}")
    with pytest.raises(LLMBudgetExceeded):
        llm.complete("one too many")


def test_extract_validates_schema(llm, backend):
    backend.responses = ['{"name": "pricing", "severity": 2}']
    assert llm.extract("x", schema=Item) == Item(name="pricing", severity=2)
    assert backend.calls[0]["schema"] is Item
    assert backend.calls[0]["model"] == "bulk-model"


def test_extract_retries_once_with_error_feedback(llm, backend):
    backend.responses = ['{"name": "pricing"}', '{"name": "pricing", "severity": 3}']
    assert llm.extract("x", schema=Item).severity == 3
    assert "did not match the required JSON schema" in backend.calls[1]["prompt"]


def test_extract_gives_up_after_retry(llm, backend):
    backend.responses = ["not json", "still not json"]
    with pytest.raises(LLMOutputError):
        llm.extract("x", schema=Item)


def test_retryable_errors_back_off_then_succeed(settings):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    backend = FakeBackend([TransientError(), TransientError(), "done"])
    llm = LLM(settings, backends={"fake": backend}, sleep=sleep, clock=lambda: now[0])
    assert llm.complete("x") == "done"
    assert sleeps == [2.0, 4.0]


def test_non_retryable_errors_propagate(settings):
    backend = FakeBackend([ValueError("bad request")])
    llm = LLM(settings, backends={"fake": backend}, sleep=lambda s: None)
    with pytest.raises(ValueError):
        llm.complete("x")


def _overloaded_smart_model():
    """A backend whose smart model always answers 503-like errors and whose bulk model works."""

    class Backend(FakeBackend):
        def generate(self, model, req):
            if model == "smart-model":
                self.calls.append({"model": model, "prompt": req.prompt})
                raise TransientError()
            return super().generate(model, req)

    return Backend(responder=lambda req: '{"name": "pricing", "severity": 2}')


def test_unavailable_smart_model_falls_back_to_bulk(settings):
    backend = _overloaded_smart_model()
    llm = LLM(settings, backends={"fake": backend}, sleep=lambda s: None)
    assert llm.extract("x", schema=Item, model="smart") == Item(name="pricing", severity=2)
    assert [c["model"] for c in backend.calls] == ["smart-model"] * 5 + ["bulk-model"]
    assert llm.usage.fallbacks == 1
    assert llm.usage.as_dict()["fallbacks"] == 1


def test_offline_falls_back_to_the_cached_bulk_answer(settings):
    live = LLM(settings, backends={"fake": _overloaded_smart_model()}, sleep=lambda s: 0)
    live.extract("x", schema=Item, model="smart")
    offline = LLM(settings, offline=True, backends={"fake": FakeBackend()})
    assert offline.extract("x", schema=Item, model="smart").severity == 2


def test_unavailable_bulk_model_is_not_hidden(settings):
    backend = FakeBackend([TransientError()] * 5)
    llm = LLM(settings, backends={"fake": backend}, sleep=lambda s: None)
    with pytest.raises(LLMUnavailable):
        llm.complete("x", model="bulk")


def test_non_retryable_smart_errors_do_not_fall_back(settings):
    backend = FakeBackend([ValueError("bad request")])
    llm = LLM(settings, backends={"fake": backend}, sleep=lambda s: None)
    with pytest.raises(ValueError):
        llm.complete("x", model="smart")
    assert llm.usage.fallbacks == 0


def test_throttle_spaces_requests(settings):
    from dataclasses import replace

    now = [0.0]
    sleeps: list[float] = []
    llm = LLM(
        replace(settings, llm_rpm=60),
        backends={"fake": FakeBackend()},
        sleep=sleeps.append,
        clock=lambda: now[0],
    )
    llm.complete("a")
    now[0] = 0.25
    llm.complete("b")
    assert sleeps == [pytest.approx(0.75)]


def test_missing_model_setting_explains_fix(settings):
    from dataclasses import replace

    llm = LLM(replace(settings, model_bulk=""), backends={"fake": FakeBackend()})
    with pytest.raises(LLMConfigError, match="MIA_MODEL_BULK"):
        llm.complete("x", model="bulk")


def test_unknown_provider(llm):
    with pytest.raises(LLMConfigError, match="No LLM backend"):
        llm.complete("x", model="nope/some-model")


def test_thinking_defaults_per_alias_and_override(llm, backend):
    llm.complete("a", model="bulk")
    llm.complete("b", model="smart")
    llm.complete("c", model="smart", thinking="high")
    assert [c["thinking"] for c in backend.calls] == ["low", "medium", "high"]


def test_full_model_id_uses_model_default_thinking(llm, backend):
    llm.complete("a", model="fake/other-model")
    assert backend.calls[0]["thinking"] is None


def test_thinking_level_is_part_of_the_cache_key(llm, backend):
    backend.responses = ["low answer", "high answer"]
    assert llm.complete("same", model="smart", thinking="low") == "low answer"
    assert llm.complete("same", model="smart", thinking="high") == "high answer"


def test_invalid_thinking_level_is_rejected(llm):
    with pytest.raises(LLMConfigError, match="Thinking level"):
        llm.complete("x", thinking="extreme")
