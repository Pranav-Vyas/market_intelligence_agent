"""The single entry point for every LLM call in the project.

All calls get: a disk cache (reruns are free and work offline), a per-run request budget,
a requests-per-minute throttle, and retries with backoff on rate limits and server errors.

Model strings are "<provider>/<model-id>". Only the Gemini backend exists today; supporting
another provider means adding a backend class here, not changing any caller.

Never call a provider SDK directly from other modules. Use `llm()` from `mia.runtime`:

    from mia.runtime import llm
    text = llm().complete("Summarize ...", model="smart")
    result = llm().extract("Extract ...", schema=MySchema)   # validated Pydantic instance
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from mia.config import Settings

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 5
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


@dataclass
class RawResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel] | None,
        temperature: float,
    ) -> RawResponse: ...

    def is_retryable(self, exc: Exception) -> bool: ...


class GeminiBackend:
    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, model, prompt, system, schema, temperature) -> RawResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
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
    by_model: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests_by_model": dict(self.by_model),
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

    # ---- public API -------------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "smart",
        temperature: float = 0.0,
    ) -> str:
        """Plain text completion. `model` is "bulk", "smart" or a full "provider/model-id"."""
        return self._call(prompt, system, None, self._resolve(model), temperature)

    def extract(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        model: str = "bulk",
        temperature: float = 0.0,
    ) -> T:
        """Structured output, validated against `schema`. Retries once with the error message."""
        full_model = self._resolve(model)
        text = self._call(prompt, system, schema, full_model, temperature)
        try:
            return schema.model_validate_json(text)
        except ValidationError as err:
            retry_prompt = (
                f"{prompt}\n\nYour previous answer did not match the required JSON schema:\n"
                f"{err}\nReturn only valid JSON matching the schema."
            )
            text = self._call(retry_prompt, system, schema, full_model, temperature)
            try:
                return schema.model_validate_json(text)
            except ValidationError as err2:
                raise LLMOutputError(f"{schema.__name__}: {err2}") from err2

    # ---- internals --------------------------------------------------------------------------

    def _resolve(self, model: str) -> str:
        if model in ("bulk", "smart"):
            env_name = f"MIA_MODEL_{model.upper()}"
            model = self.settings.model_bulk if model == "bulk" else self.settings.model_smart
            if not model:
                raise LLMConfigError(
                    f"{env_name} is not set. Run `uv run mia models` to list the model IDs your "
                    f"key can use, then set {env_name}=gemini/<model-id> in .env."
                )
        if "/" not in model:
            raise LLMConfigError(f"Model {model!r} must look like 'provider/model-id'.")
        return model

    def _backend(self, provider: str) -> Backend:
        if provider not in self._backends:
            if provider == "gemini":
                self._backends[provider] = GeminiBackend(self.settings.gemini_api_key)
            else:
                raise LLMConfigError(f"No LLM backend for provider {provider!r}.")
        return self._backends[provider]

    def _cache_path(self, model, prompt, system, schema, temperature) -> Path:
        key = json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "schema": schema.model_json_schema() if schema else None,
                "temperature": temperature,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _call(self, prompt, system, schema, model, temperature) -> str:
        path = self._cache_path(model, prompt, system, schema, temperature)
        if path.exists():
            self.usage.cache_hits += 1
            return json.loads(path.read_text())["text"]
        if self.offline:
            raise LLMCacheMiss(f"No cached response for this {model} call (offline mode).")
        if self.usage.requests >= self.settings.max_llm_calls_per_run:
            raise LLMBudgetExceeded(
                f"Reached MIA_MAX_LLM_CALLS_PER_RUN={self.settings.max_llm_calls_per_run}."
            )

        provider, model_id = model.split("/", 1)
        backend = self._backend(provider)
        raw = self._generate_with_retries(
            backend, model, model_id, prompt, system, schema, temperature
        )

        self.usage.requests += 1
        self.usage.input_tokens += raw.input_tokens
        self.usage.output_tokens += raw.output_tokens
        self.usage.by_model[model] = self.usage.by_model.get(model, 0) + 1

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "text": raw.text,
                    "input_tokens": raw.input_tokens,
                    "output_tokens": raw.output_tokens,
                }
            )
        )
        return raw.text

    def _generate_with_retries(
        self, backend, model, model_id, prompt, system, schema, temperature
    ) -> RawResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle(model)
            try:
                return backend.generate(model_id, prompt, system, schema, temperature)
            except Exception as exc:
                if attempt == MAX_ATTEMPTS or not backend.is_retryable(exc):
                    raise
                self._sleep(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
        raise AssertionError("unreachable")

    def _throttle(self, model: str) -> None:
        min_interval = 60.0 / max(self.settings.llm_rpm, 1)
        last = self._last_call.get(model)
        if last is not None:
            wait = min_interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[model] = self._clock()
