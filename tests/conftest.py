from __future__ import annotations

import hashlib

import pytest

from mia import runtime
from mia.config import Settings
from mia.llm import LLM, LLMRequest, RawResponse


class FakeBackend:
    """Stands in for a provider. Returns queued responses and records every call."""

    def __init__(self, responses: list[str | Exception] | None = None, responder=None):
        """`responder(req) -> str` computes a reply when the queue of `responses` is empty."""
        self.responses = list(responses or [])
        self.responder = responder
        self.calls: list[dict] = []

    def generate(self, model, req: LLMRequest) -> RawResponse:
        self.calls.append(
            {
                "model": model,
                "prompt": req.prompt,
                "system": req.system,
                "schema": req.schema,
                "thinking": req.thinking,
            }
        )
        if self.responses:
            item = self.responses.pop(0)
        else:
            item = self.responder(req) if self.responder else "ok"
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
        thinking_bulk="low",
        thinking_smart="medium",
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


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words are close. No model download."""

    name = "fake-embedder"
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        from mia.rag.bm25 import tokenize

        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[int(hashlib.sha1(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]


class FakeReranker:
    """Scores a text by how many query words it contains."""

    def scores(self, query: str, texts: list[str]) -> list[float]:
        from mia.rag.bm25 import tokenize

        terms = set(tokenize(query))
        return [float(len(terms & set(tokenize(t)))) for t in texts]


@pytest.fixture(autouse=True)
def _local_models(monkeypatch, tmp_path, request):
    """Retrieval never downloads a model in tests, and its embedding cache stays in tmp_path.
    Tests marked `network` use the real models."""
    if "network" in request.keywords:
        return
    from mia.rag import dense, rerank

    monkeypatch.setattr(dense, "default_embedder", FakeEmbedder)
    monkeypatch.setattr(rerank, "default_reranker", FakeReranker)
    if runtime._settings is None:
        monkeypatch.setattr(runtime, "_settings", Settings(data_dir=tmp_path / "data"))


@pytest.fixture
def offline_world(monkeypatch):
    """Point the pipeline's network-bound stages at the fixture documents (Noted, Scribely,
    Recapper), so end-to-end tests never touch the internet."""
    from mia import pipeline
    from mia.schemas import Competitor
    from tests.fixtures import load_documents

    docs = load_documents()
    competitors = [
        Competitor(name="Noted", website="https://noted.example.com"),
        Competitor(name="Scribely", website="https://scribely.example.com"),
        Competitor(name="Recapper", website="https://recapper.example.com"),
    ]
    monkeypatch.setattr(pipeline, "discover_competitors", lambda market, k=8: competitors[:k])
    monkeypatch.setattr(
        pipeline,
        "collect_official",
        lambda c: [
            d for d in docs if d.company == c.name and d.source_type in ("official", "pricing")
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "collect_complaints",
        lambda c, market, source, query=None, limit=40: [
            d for d in docs if d.company == c.name and d.source_type == source
        ][:limit],
    )
    return {"competitors": competitors, "documents": docs}
