from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mia import pipeline, runtime
from mia.agent.customer_scout import scout
from mia.llm import LLM
from mia.schemas import Competitor, Document, doc_id
from tests.conftest import FakeBackend

ACME = Competitor(name="Acme", website="https://acme.example.com")
MARKET = "AI meeting assistants"


def _docs(source: str, prefix: str, n: int) -> list[Document]:
    return [
        Document(
            id=doc_id(f"https://{source}.example.com/{prefix}{i}"),
            url=f"https://{source}.example.com/{prefix}{i}",
            source_type=source,
            company="Acme",
            text=f"{prefix} complaint number {i}",
        )
        for i in range(n)
    ]


class World:
    """Fake collectors: default queries find a few documents, queries find more."""

    def __init__(self, default: dict[str, int], per_query: int = 5):
        self.default, self.per_query, self.calls = default, per_query, []

    def collect(self, c, market, source, query=None, limit=40):
        self.calls.append((source, query))
        if source == "app_store" and query == "broken site":
            raise RuntimeError("HTTP 503 from the store")
        if query is None:
            return _docs(source, "default", self.default.get(source, 0))
        return _docs(source, query.replace(" ", "-"), self.per_query)


def _actions(*actions: dict) -> list[str]:
    return [json.dumps(a) for a in actions]


@pytest.fixture
def setup(settings, monkeypatch):
    def install(world: World, replies: list[str], **overrides):
        s = replace(settings, **{"max_llm_calls_per_run": 50, **overrides})
        backend = FakeBackend(replies, responder=lambda req: json.dumps({"action": "done"}))
        runtime.configure(settings=s, llm_client=LLM(s, backends={"fake": backend}))
        monkeypatch.setattr(pipeline, "collect_complaints", world.collect)
        return backend

    return install


def test_enough_documents_from_default_searches_means_no_llm_call(setup):
    world = World({"hn": 10, "app_store": 8})
    backend = setup(world, [])
    docs, steps = scout(ACME, MARKET)
    assert len(docs) == 18
    assert world.calls == [("hn", None), ("app_store", None)]
    assert backend.calls == []
    assert steps[-1]["action"] == "done" and "reached 18" in steps[-1]["result_summary"]


def test_scout_searches_until_the_threshold(setup):
    world = World({"hn": 3, "app_store": 0})
    backend = setup(
        world,
        _actions(
            {"action": "search", "source": "hn", "query": "bot joins calls", "reason": "r"},
            {"action": "search", "source": "app_store", "query": "crash", "reason": "r"},
            {"action": "search", "source": "hn", "query": "cancel", "reason": "r"},
        ),
    )
    docs, steps = scout(ACME, MARKET)
    assert len(docs) == 18  # 3 + 5 + 5 + 5
    assert world.calls[2:] == [("hn", "bot joins calls"), ("app_store", "crash"), ("hn", "cancel")]
    assert [c["model"] for c in backend.calls] == ["smart-model"] * 3
    assert "Tried so far:\n- hn default queries: 3 new" in backend.calls[0]["prompt"]
    assert steps[-1]["result_summary"] == "reached 18 complaint documents"
    assert sum(step["requests"] for step in steps) == 3


def test_a_failing_collector_is_data_not_a_crash(setup):
    world = World({"hn": 1})
    backend = setup(
        world,
        _actions(
            {"action": "search", "source": "app_store", "query": "broken site", "reason": "r"},
            {"action": "done", "reason": "nothing more to find"},
        ),
    )
    docs, steps = scout(ACME, MARKET)
    assert len(docs) == 1
    assert "failed: RuntimeError: HTTP 503 from the store" in steps[2]["result_summary"]
    assert "(failed: RuntimeError" in backend.calls[1]["prompt"]  # the LLM is told
    assert steps[-1]["result_summary"] == "scout chose to stop: nothing more to find"


def test_repeated_or_invalid_actions_use_a_step_and_are_skipped(setup):
    world = World({"hn": 0, "app_store": 0})
    setup(
        world,
        _actions(
            {"action": "search", "source": "hn", "query": "slow"},
            {"action": "search", "source": "hn", "query": "slow"},  # repeated
            {"action": "search", "source": "twitter", "query": "x"},  # unknown source
            {"action": "search", "source": "hn", "query": "  "},  # empty
        ),
    )
    docs, steps = scout(ACME, MARKET)
    assert [s["action"] for s in steps] == [
        "search",
        "search",
        "search",
        "invalid",
        "invalid",
        "invalid",
        "stop",
    ]
    assert "step cap reached with 5 complaint documents (target 15)" in steps[-1]["result_summary"]
    assert len(docs) == 5


def test_focus_skips_default_searches_and_reports_new_documents(setup):
    world = World({"hn": 30, "app_store": 30})
    known = frozenset(d.id for d in _docs("hn", "default", 30))
    setup(
        world,
        _actions(
            {"action": "search", "source": "hn", "query": "calendar duplicates"},
            {"action": "search", "source": "app_store", "query": "calendar"},
        ),
    )
    docs, steps = scout(ACME, MARKET, focus="duplicate calendar events", known=known)
    assert all(q is not None for _, q in world.calls)
    assert len(docs) == 10
    assert "targeted research on 'duplicate calendar events': 10 new" in steps[-1]["result_summary"]


def test_collection_budget_running_out_stops_the_scout(setup):
    world = World({"hn": 2})
    setup(world, [], max_llm_calls_per_run=1, scout_budget_share=0.6)  # collection share: 0
    docs, steps = scout(ACME, MARKET)
    assert len(docs) == 2
    assert steps[-1]["action"] == "stop"
    assert "collection phase has used its 0 requests" in steps[-1]["result_summary"]
