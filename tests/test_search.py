from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from mia import http, runtime
from mia.collectors import search as search_mod
from mia.collectors.access import AccessResult, as_markdown
from mia.collectors.search import search


class FakeDDGS:
    calls = 0
    rows: list[dict] = []

    def text(self, query, max_results):
        FakeDDGS.calls += 1
        return FakeDDGS.rows[:max_results]


@pytest.fixture
def fake_ddgs(monkeypatch):
    import ddgs

    FakeDDGS.calls = 0
    FakeDDGS.rows = [
        {"href": "https://a.example.com", "title": "A", "body": "about a"},
        {"href": "https://a.example.com/", "title": "A again", "body": "duplicate"},
        {"href": "https://b.example.com", "title": "B", "body": "about b"},
    ]
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    return FakeDDGS


@pytest.fixture
def use_http(monkeypatch):
    """Point the module-level HTTP client at a fake transport."""

    def install(handler):
        client = http.HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
        monkeypatch.setattr(http, "_default", client)

    return install


def test_without_tavily_key_uses_ddgs_and_dedupes(settings, fake_ddgs):
    runtime.configure(settings=settings)
    results = search("otter alternatives", k=5)
    assert [r.url for r in results] == ["https://a.example.com", "https://b.example.com"]
    assert results[0].snippet == "about a"


def test_ddgs_results_are_cached(settings, fake_ddgs):
    runtime.configure(settings=settings)
    search("same query", k=5)
    search("same query", k=5)
    assert fake_ddgs.calls == 1


def test_empty_ddgs_results_are_not_cached(settings, fake_ddgs):
    runtime.configure(settings=settings)
    fake_ddgs.rows = []
    assert search("nothing", k=5) == []
    search("nothing", k=5)
    assert fake_ddgs.calls == 2


def test_tavily_is_used_when_key_is_set(settings, fake_ddgs, use_http):
    runtime.configure(settings=replace(settings, tavily_api_key="tvly-test"))
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(
            200, json={"results": [{"url": "https://t.example.com", "title": "T", "content": "c"}]}
        )

    use_http(handler)
    results = search("q", k=3)
    assert [r.url for r in results] == ["https://t.example.com"]
    assert seen["auth"] == "Bearer tvly-test"
    assert fake_ddgs.calls == 0


def test_tavily_failure_falls_back_to_ddgs(settings, fake_ddgs, use_http):
    runtime.configure(settings=replace(settings, tavily_api_key="tvly-test"))
    use_http(lambda request: httpx.Response(401, json={"detail": "bad key"}))
    results = search("q", k=3)
    assert results and fake_ddgs.calls == 1


def test_dedupe_ignores_trailing_slash_and_fragment():
    from mia.schemas import SearchResult

    rows = [
        SearchResult(url="https://x.com/a", title="", snippet=""),
        SearchResult(url="https://x.com/a/", title="", snippet=""),
        SearchResult(url="https://x.com/a#top", title="", snippet=""),
    ]
    assert len(search_mod._dedupe(rows)) == 1


def test_access_table_is_markdown():
    table = as_markdown([AccessResult("HN", "works", "3 | 4 hits")])
    assert table.splitlines()[2] == "| HN | works | 3 / 4 hits |"
