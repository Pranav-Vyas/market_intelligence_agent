"""Product Scout and Discovery Scout decisions (task A13)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mia import runtime
from mia.agent import discovery_scout, product_scout
from mia.collectors import web
from mia.llm import COLLECTION, LLM
from mia.schemas import Competitor
from tests.conftest import FakeBackend

ACME = Competitor(name="Acme", website="https://acme.example.com")
LINKS = [
    ("https://acme.example.com/features", "Features"),
    ("https://acme.example.com/plans", "Plans"),
    ("https://acme.example.com/customers", "Customer stories"),
    ("https://acme.example.com/security", "Trust center"),
]


@pytest.fixture
def use(settings):
    def install(*replies: str, **overrides):
        s = replace(settings, **{"max_llm_calls_per_run": 20, **overrides})
        backend = FakeBackend(list(replies))
        runtime.configure(settings=s, llm_client=LLM(s, backends={"fake": backend}))
        return backend

    return install


def _picks(*pages: tuple[int, str]) -> str:
    return json.dumps({"pages": [{"link_id": i, "kind": k} for i, k in pages]})


def test_product_scout_picks_pricing_and_other_pages(use):
    backend = use(_picks((1, "pricing"), (0, "features"), (3, "security"), (9, "other")))
    pricing, others = product_scout.choose_pages(ACME, LINKS)
    assert pricing == "https://acme.example.com/plans"
    assert others == ["https://acme.example.com/features", "https://acme.example.com/security"]
    assert "[2] https://acme.example.com/customers — Customer stories" in backend.calls[0]["prompt"]
    assert backend.calls[0]["model"] == "smart-model"


def test_product_scout_respects_the_page_cap(use):
    use(_picks(*[(i, "features") for i in range(4)]), max_product_pages=4)
    _, others = product_scout.choose_pages(ACME, LINKS)
    assert len(others) == 2  # 4 pages minus the homepage and the pricing page


def test_product_scout_keeps_the_keyword_choice_when_the_llm_fails(use):
    use("not json", "still not json")
    assert product_scout.choose_pages(ACME, LINKS) is None
    use(_picks())
    assert product_scout.choose_pages(ACME, LINKS) is None
    assert product_scout.choose_pages(ACME, []) is None


def test_collect_official_fetches_the_chosen_pages(settings, monkeypatch):
    runtime.configure(settings=settings)
    home = "".join(f'<a href="{u}">{t}</a>' for u, t in LINKS) + '<a href="/blog/x">Blog</a>'
    fetched = []

    def fake_fetch(url, source_type, company):
        fetched.append((url, source_type))
        from mia.schemas import Document, doc_id

        doc = Document(id=doc_id(url), url=url, source_type=source_type, company=company, text="t")
        return doc, home if url == ACME.website else ""

    monkeypatch.setattr(web, "_fetch", fake_fetch)
    seen_links = []

    def choose(c, links, keyword):
        seen_links.extend(links)
        return "https://acme.example.com/plans", ["https://acme.example.com/security"]

    web.collect_official(ACME, choose=choose)
    assert ("https://acme.example.com/plans", "pricing") in fetched
    assert ("https://acme.example.com/security", "official") in fetched
    assert ("https://acme.example.com/features", "official") not in fetched
    assert all("/blog/" not in u for u, _ in seen_links)  # skipped before the LLM sees them


def test_discovery_scout_proposes_new_queries_only(use):
    reply = {"queries": ["best ai notetaker", "Otter alternatives", "  "], "reason": "more lists"}
    backend = use(json.dumps(reply))
    queries, how = discovery_scout.next_queries(
        "AI meeting assistants", [ACME], ["otter alternatives", "best ai meeting assistants"]
    )
    assert queries == ["best ai notetaker"]  # the repeated query and the empty one are dropped
    assert how == "LLM queries (more lists)"
    assert "Products validated so far: Acme" in backend.calls[0]["prompt"]


def test_discovery_scout_falls_back_to_alternatives_queries(use):
    use("not json", "still not json")
    queries, how = discovery_scout.next_queries("AI meeting assistants", [ACME], [])
    assert queries == ["Acme alternatives"]
    assert how == "fallback alternatives queries"


def test_scout_decisions_count_as_collection(use):
    backend = use(_picks((1, "pricing")))
    llm = runtime.llm()
    with llm.phase(COLLECTION):
        product_scout.choose_pages(ACME, LINKS)
    assert llm.usage.by_phase == {"collection": 1}
    assert len(backend.calls) == 1
