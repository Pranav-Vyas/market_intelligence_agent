"""Web search (task B2): Tavily when TAVILY_API_KEY is set, `ddgs` (no key) otherwise or as a
fallback when Tavily fails. Results are cached, so repeated queries cost no credits."""

from __future__ import annotations

import logging

from mia import cache, http, runtime
from mia.schemas import SearchResult

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


class SearchError(Exception):
    pass


def search(query: str, k: int = 10) -> list[SearchResult]:
    key = runtime.settings().tavily_api_key
    if key:
        try:
            results = _tavily(query, k, key)
            if results:
                return results
        except (http.FetchError, SearchError, KeyError, ValueError) as err:
            log.warning("Tavily search failed (%s); falling back to ddgs", err)
    return _ddgs(query, k)


def _tavily(query: str, k: int, api_key: str) -> list[SearchResult]:
    resp = http.post_json(
        TAVILY_URL,
        {"query": query, "max_results": k, "search_depth": "basic"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not resp.ok:
        raise SearchError(f"Tavily returned HTTP {resp.status}")
    results = [
        SearchResult(url=r["url"], title=r.get("title", ""), snippet=r.get("content", ""))
        for r in resp.json().get("results", [])
    ]
    return _dedupe(results)[:k]


def _ddgs(query: str, k: int) -> list[SearchResult]:
    def fetch() -> list[dict]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        try:
            raw = DDGS().text(query, max_results=k)
        except DDGSException as err:
            raise SearchError(f"ddgs search failed: {err}") from err
        return [
            {"url": r["href"], "title": r.get("title", ""), "snippet": r.get("body", "")}
            for r in raw
            if r.get("href")
        ] or None  # don't cache an empty result

    rows = cache.get_or_fetch("ddgs", {"query": query, "k": k}, fetch) or []
    return _dedupe([SearchResult(**r) for r in rows])[:k]


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique = []
    for r in results:
        key = r.url.rstrip("/").split("#")[0]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
