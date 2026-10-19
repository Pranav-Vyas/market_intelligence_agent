"""The single entry point for every LLM call in the project.

All calls get: a disk cache (reruns are free and work offline), a per-run request budget,
a requests-per-minute throttle, and retries with backoff on rate limits and server errors.

Model strings are "<provider>/<model-id>". Only the Gemini backend exists today; supporting
another provider means adding a backend class here, not changing any caller.

Never call a provider SDK directly from other modules. Use `llm()` from `mia.runtime`:

    from mia.runtime import llm
    text = llm().complete("Summarize ...", model="smart")
    result = llm().extract("Extract ...", schema=MySchema)   # validated Pydantic instance

Fallback: when the smart model stays unavailable after every retry (or, offline, has no cached
answer), the call is made with the bulk model instead and `usage.fallbacks` counts it. Free-tier
Flash models are often overloaded; Flash-Lite rarely is.

Thinking: Gemini 3 models think before answering. Each alias has a default level from .env
(MIA_THINKING_BULK, MIA_THINKING_SMART); pass `thinking="minimal" | "low" | "medium" | "high"` to
override it for one call. Not every model supports every level (gemini-3.8-flash rejects
"minimal"); Gemini returns a clear 400 error in that case, which is not retried.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from mia.config import Settings
from mia.jsonfile import write_json

T = TypeVar("T", bound=BaseModel)
COLLECTION = "collection"
R = TypeVar("R")
log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
THINKING_LEVELS = ("minimal", "low", "medium", "high")
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0


class LLMError(Exception):
    pass


class LLMConfigError(LLMError):
    pass


class LLMCacheMiss(LLMError):
    """Offline mode and the response isn't in the cache."""


class LLMBudgetExceeded(LLMError):
    """The run hit MIA_MAX_LLM_CALLS_PER_RUN."""


class LLMOutputError(LLMError):
    """The model returned output that doesn't match the requested schema, even after a retry."""


class LLMUnavailable(LLMError):
    """Rate limits or server errors persisted through every retry (e.g. 503 "high demand")."""


# What a stage can survive by skipping one item. LLMConfigError is not here: it needs a fix.
RECOVERABLE = (LLMOutputError, LLMCacheMiss, LLMUnavailable, LLMBudgetExceeded)


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    schema: type[BaseModel] | None = None
    temperature: float = 0.0
    thinking: str | None = None  # one of THINKING_LEVELS; None = the model's default


@dataclass
class RawResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(Protocol):
    def generate(self, model: str, req: LLMRequest) -> RawResponse: ...

    def is_retryable(self, exc: Exception) -> bool: ...


class GeminiBackend:
    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, model: str, req: LLMRequest) -> RawResponse:
        from google.genai import types

        thinking = None
        if req.thinking:
            level = types.ThinkingLevel(req.thinking.upper())
            thinking = types.ThinkingConfig(thinking_level=level)
        config = types.GenerateContentConfig(
            system_instruction=req.system,
            temperature=req.temperature,
            response_mime_type="application/json" if req.schema else None,
            response_schema=req.schema,
            thinking_config=thinking,
        )
        resp = self._client.models.generate_content(model=model, contents=req.prompt, config=config)
        usage = resp.usage_metadata
        output_tokens = 0
        if usage:
            output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return RawResponse(
            text=resp.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=output_tokens,
        )

    def is_retryable(self, exc: Exception) -> bool:
        from google.genai import errors

        return isinstance(exc, errors.APIError) and exc.code in (429, 500, 502, 503, 504)

    def list_models(self) -> list[str]:
        names = []
        for m in self._client.models.list():
            if "generateContent" in (m.supported_actions or []):
                names.append((m.name or "").removeprefix("models/"))
        return sorted(names)


@dataclass
class Usage:
    requests: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fallbacks: int = 0  # smart-model calls answered by the bulk model
    by_model: dict[str, int] = field(default_factory=dict)
    by_phase: dict[str, int] = field(default_factory=dict)  # e.g. {"collection": 12}

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallbacks": self.fallbacks,
            "requests_by_model": dict(self.by_model),
            "requests_by_phase": dict(self.by_phase),
        }


class LLM:
    def __init__(
        self,
        settings: Settings,
        *,
        offline: bool = False,
        backends: dict[str, Backend] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.offline = offline
        self.usage = Usage()
        self._backends: dict[str, Backend] = dict(backends or {})
        self._sleep = sleep
        self._clock = clock
        self._last_call: dict[str, float] = {}
        self._cache_dir = settings.cache_dir / "llm"
        self._lock = threading.Lock()  # usage, budget and throttle are shared by graph branches
        self._per_thread: dict[int, list[int]] = {}
        self._local = threading.local()  # the calling thread's phase

    # ---- public API -------------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "smart",
        temperature: float = 0.0,
        thinking: str | None = None,
    ) -> str:
        """Plain text completion. `model` is "bulk", "smart" or a full "provider/model-id"."""

        def run(full_model: str, level: str | None) -> str:
            return self._call(full_model, LLMRequest(prompt, system, None, temperature, level))

        return self._with_fallback(model, thinking, run)

    def extract(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        model: str = "bulk",
        temperature: float = 0.0,
        thinking: str | None = None,
    ) -> T:
        """Structured output, validated against `schema`. Retries once with the error message."""

        def run(full_model: str, level: str | None) -> T:
            text = self._call(full_model, LLMRequest(prompt, system, schema, temperature, level))
            try:
                return schema.model_validate_json(text)
            except ValidationError as err:
                retry_prompt = (
                    f"{prompt}\n\nYour previous answer did not match the required JSON schema:\n"
                    f"{err}\nReturn only valid JSON matching the schema."
                )
                retry = LLMRequest(retry_prompt, system, schema, temperature, level)
                text = self._call(full_model, retry)
                try:
                    return schema.model_validate_json(text)
                except ValidationError as err2:
                    raise LLMOutputError(f"{schema.__name__}: {err2}") from err2

        return self._with_fallback(model, thinking, run)

    # ---- internals --------------------------------------------------------------------------

    def _with_fallback(
        self, model: str, thinking: str | None, run: Callable[[str, str | None], R]
    ) -> R:
        full_model, default_thinking = self._resolve(model)
        try:
            return run(full_model, self._thinking(thinking, default_thinking))
        except (LLMUnavailable, LLMCacheMiss) as err:
            bulk = self.settings.model_bulk
            if model != "smart" or not bulk or bulk == full_model:
                raise
            log.warning("%s is unavailable (%s); using the bulk model %s", full_model, err, bulk)
            self.usage.fallbacks += 1
            bulk_model, bulk_thinking = self._resolve("bulk")
            return run(bulk_model, self._thinking(thinking, bulk_thinking))

    def _resolve(self, model: str) -> tuple[str, str | None]:
        """Return the full model string and its default thinking level."""
        default_thinking = None
        if model in ("bulk", "smart"):
            env_name = f"MIA_MODEL_{model.upper()}"
            s = self.settings
            model, default_thinking = (
                (s.model_bulk, s.thinking_bulk)
                if model == "bulk"
                else (s.model_smart, s.thinking_smart)
            )
            if not model:
                raise LLMConfigError(
                    f"{env_name} is not set. Run `uv run mia models` to list the model IDs your "
                    f"key can use, then set {env_name}=gemini/<model-id> in .env."
                )
        if "/" not in model:
            raise LLMConfigError(f"Model {model!r} must look like 'provider/model-id'.")
        return model, default_thinking or None

    @staticmethod
    def _thinking(requested: str | None, default: str | None) -> str | None:
        level = requested or default
        if level is not None and level not in THINKING_LEVELS:
            raise LLMConfigError(f"Thinking level {level!r} must be one of {THINKING_LEVELS}.")
        return level

    def _backend(self, provider: str) -> Backend:
        if provider not in self._backends:
            if provider == "gemini":
                self._backends[provider] = GeminiBackend(self.settings.gemini_api_key)
            else:
                raise LLMConfigError(f"No LLM backend for provider {provider!r}.")
        return self._backends[provider]

    def _cache_path(self, model: str, req: LLMRequest) -> Path:
        key = json.dumps(
            {
                "model": model,
                "system": req.system,
                "prompt": req.prompt,
                "schema": req.schema.model_json_schema() if req.schema else None,
                "temperature": req.temperature,
                "thinking": req.thinking,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _call(self, model: str, req: LLMRequest) -> str:
        path = self._cache_path(model, req)
        if path.exists():
            with self._lock:
                self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        if self.offline:
            raise LLMCacheMiss(f"No cached response for this {model} call (offline mode).")
        phase = getattr(self._local, "phase", None)
        with self._lock:  # reserve the request first, so parallel branches can't overshoot
            if self.usage.requests >= self.settings.max_llm_calls_per_run:
                raise LLMBudgetExceeded(
                    f"Reached MIA_MAX_LLM_CALLS_PER_RUN={self.settings.max_llm_calls_per_run}."
                )
            limit = self._phase_limit(phase)
            if limit is not None and self.usage.by_phase.get(phase, 0) >= limit:
                raise LLMBudgetExceeded(f"The {phase} phase has used its {limit} requests.")
            self.usage.requests += 1
            if phase:
                self.usage.by_phase[phase] = self.usage.by_phase.get(phase, 0) + 1

        provider, model_id = model.split("/", 1)
        try:
            raw = self._generate_with_retries(self._backend(provider), model, model_id, req)
        except BaseException:
            with self._lock:
                self.usage.requests -= 1
                if phase:
                    self.usage.by_phase[phase] -= 1
            raise

        with self._lock:
            self.usage.input_tokens += raw.input_tokens
            self.usage.output_tokens += raw.output_tokens
            self.usage.by_model[model] = self.usage.by_model.get(model, 0) + 1
            mine = self._per_thread.setdefault(threading.get_ident(), [0, 0])
            mine[0] += 1
            mine[1] += raw.input_tokens + raw.output_tokens
        record = {
            "model": model,
            "text": raw.text,
            "input_tokens": raw.input_tokens,
            "output_tokens": raw.output_tokens,
        }
        write_json(path, record)
        return raw.text

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Count this thread's calls towards a phase. "collection" (the scouts and discovery)
        may use at most `scout_budget_share` of the request cap, so the Analyst and Verifier
        always have requests left (SPEC.md §3)."""
        previous = getattr(self._local, "phase", None)
        self._local.phase = name
        try:
            yield
        finally:
            self._local.phase = previous

    def phase_left(self, name: str) -> int | None:
        """Requests the phase may still make, or None if it has no share of its own."""
        limit = self._phase_limit(name)
        if limit is None:
            return None
        with self._lock:
            return max(0, limit - self.usage.by_phase.get(name, 0))

    def _phase_limit(self, phase: str | None) -> int | None:
        if phase != COLLECTION:
            return None
        return int(self.settings.max_llm_calls_per_run * self.settings.scout_budget_share)

    def thread_usage(self) -> tuple[int, int]:
        """(requests, tokens) made so far by the calling thread: one graph branch's share."""
        with self._lock:
            requests, tokens = self._per_thread.get(threading.get_ident(), [0, 0])
        return requests, tokens

    def _generate_with_retries(
        self, backend: Backend, model: str, model_id: str, req: LLMRequest
    ) -> RawResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle(model)
            try:
                return backend.generate(model_id, req)
            except Exception as exc:
                if not backend.is_retryable(exc):
                    raise
                if attempt == MAX_ATTEMPTS:
                    raise LLMUnavailable(f"{model}: {exc}") from exc
                self._sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
        raise AssertionError("unreachable")

    def _throttle(self, model: str) -> None:
        """Wait for this model's next free slot. Slots are handed out under the lock, so
        parallel callers queue up instead of all waiting for the same moment."""
        min_interval = 60.0 / max(self.settings.llm_rpm, 1)
        with self._lock:
            now = self._clock()
            last = self._last_call.get(model)
            start = now if last is None else max(now, last + min_interval)
            self._last_call[model] = start
        if start > now:
            self._sleep(start - now)
