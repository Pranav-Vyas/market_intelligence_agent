from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from mia import discovery, runtime
from mia.discovery import discover_competitors, name_key
from mia.schemas import Document, SearchResult, doc_id

# Fictional market. Each listicle names some products; websites as the LLM would give them.
WEBSITES = {
    "Noted": "noted.example.com",
    "Scribely": "scribely.example.com",
    "Recapper": "recapper.example.com",
    "MinuteMind": "",
    "BlogCo": "blogco.example.com",
    "EchoNotes": "echonotes.example.com",
    "Quillo": "quillo.example.com",
}
LISTICLES = {
    "https://blog.example.com/best": "Best tools: Noted, Scribely, Recapper and MinuteMind.",
    "https://news.example.com/top": "Our picks: Noted.ai, Scribely and BlogCo.",
    "https://reviews.example.com/compare": "Noted AI vs Recapper: a comparison.",
    "https://alt.example.com/noted-alternatives": "Alternatives to Noted: EchoNotes, Quillo.",
}
HOMEPAGES = {  # BlogCo is a blog, not a product; Recapper's site is down
    "https://noted.example.com": "Noted records and summarizes meetings.",
    "https://scribely.example.com": "Scribely: meeting notes for small teams.",
    "https://minutemind.example.com": "MinuteMind: multilingual meeting transcripts.",
    "https://blogco.example.com": "BlogCo: articles about productivity.",
    "https://echonotes.example.com": "EchoNotes: open-source meeting recorder.",
    "https://quillo.example.com": "Quillo: AI notes for every call.",
}


class World:
    def __init__(self):
        self.queries: list[str] = []
        self.fetched: list[str] = []

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        self.queries.append(query)
        if query.endswith("official site"):
            return [
                SearchResult(url="https://www.g2.com/products/minutemind", title="", snippet=""),
                SearchResult(url="https://minutemind.example.com/home", title="", snippet=""),
            ]
        if "alternatives" in query and query.split()[0] in ("Noted", "Scribely"):
            return [
                SearchResult(
                    url="https://alt.example.com/noted-alternatives", title="Alt", snippet=""
                )
            ]
        return [
            SearchResult(url=u, title="List", snippet="")
            for u in LISTICLES
            if "alternatives" not in u
        ] + [
            SearchResult(
                url="https://www.reddit.com/r/x/1",
                title="Reddit thread",
                snippet="I switched from Noted to Quillo",
            )
        ]

    def fetch_page(self, url, source_type, company):
        self.fetched.append(url)
        text = LISTICLES.get(url) or HOMEPAGES.get(url.rstrip("/"))
        if text is None:
            return None
        return Document(
            id=doc_id(url), url=url, source_type=source_type, company=company, text=text
        )


def respond(req) -> str:
    name = req.schema.__name__ if req.schema else ""
    if name == "_Mentions":
        out = []
        for sid, text in re.findall(r'<source id="(\d+)">(.*?)</source>', req.prompt, re.S):
            found = [p for p in WEBSITES if p in text]
            out.append(
                {
                    "source_id": int(sid),
                    "products": [{"name": p, "website": WEBSITES[p]} for p in found],
                }
            )
        return json.dumps({"sources": out})
    if name == "_Validation":
        cands = re.findall(r'<candidate id="(\d+)" name="([^"]+)"', req.prompt)
        return json.dumps(
            {
                "verdicts": [
                    {
                        "candidate_id": int(i),
                        "is_product": n != "BlogCo",
                        "one_liner": f"{n} does notes",
                    }
                    for i, n in cands
                ]
            }
        )
    return "{}"


@pytest.fixture
def world(settings, llm, backend, monkeypatch, tmp_path):
    w = World()
    monkeypatch.setattr(discovery, "search", w.search)
    monkeypatch.setattr(discovery, "fetch_page", w.fetch_page)
    monkeypatch.setattr(discovery, "REPO_ROOT", tmp_path)  # no overrides.yaml by default
    backend.responder = respond
    runtime.configure(settings=replace(settings, min_competitors=3), llm_client=llm)
    return w


def test_ranks_validated_competitors_by_mentions(world):
    found = discover_competitors("AI meeting assistants", k=8)
    names = [c.name for c in found]
    assert names[:2] == ["Noted", "Scribely"]  # 2+ independent sources each
    assert "BlogCo" not in names  # rejected by validation
    assert "Recapper" not in names  # homepage unreachable
    noted = found[0]
    assert noted.mention_count == 3  # "Noted", "Noted.ai" and "Noted AI" are one product
    assert noted.website == "https://noted.example.com"
    assert noted.one_liner == "Noted does notes"


def test_review_and_social_sites_are_not_fetched(world):
    discover_competitors("AI meeting assistants")
    assert not any("reddit.com" in u or "g2.com" in u for u in world.fetched)


def test_reddit_results_are_dropped_including_snippets(world):
    names = [c.name for c in discover_competitors("AI meeting assistants")]
    assert "Quillo" not in names  # only the Reddit snippet mentions it in round one


def test_missing_website_is_found_by_search_skipping_review_sites(world):
    found = {c.name: c for c in discover_competitors("AI meeting assistants")}
    assert found["MinuteMind"].website.startswith("https://minutemind.example.com")
    assert "MinuteMind official site" in world.queries


def test_second_round_searches_alternatives_when_too_few_validate(world, settings, backend):
    from mia.llm import LLM

    s = replace(settings, min_competitors=5, max_llm_calls_per_run=20)
    runtime.configure(settings=s, llm_client=LLM(s, backends={"fake": backend}))
    names = [c.name for c in discover_competitors("AI meeting assistants")]
    assert "Noted alternatives" in world.queries
    assert {"EchoNotes", "Quillo"} <= set(names)


def test_k_limits_the_result(world):
    assert len(discover_competitors("AI meeting assistants", k=2)) == 2


def test_overrides_replace_website_and_pricing_url(world, tmp_path):
    (tmp_path / "overrides.yaml").write_text(
        "ai-meeting-assistants:\n"
        "  noted.ai:\n"
        "    website: https://www.noted.example.com\n"
        "    pricing_url: https://www.noted.example.com/plans\n"
    )
    noted = discover_competitors("AI meeting assistants")[0]
    assert noted.website == "https://www.noted.example.com"
    assert noted.pricing_url == "https://www.noted.example.com/plans"


def test_uses_few_llm_calls(world, backend):
    discover_competitors("AI meeting assistants")
    assert len(backend.calls) <= 3  # one mentions call per 4 sources + one validation call
    assert {c["model"] for c in backend.calls} == {"bulk-model"}


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Otter.ai", "otter"),
        ("Otter AI", "otter"),
        ("otter", "otter"),
        ("tl;dv", "tldv"),
        ("Read.ai", "read"),
        ("https://www.fireflies.ai", "fireflies"),
        ("Fathom", "fathom"),
    ],
)
def test_name_key(raw, key):
    assert name_key(raw) == key
