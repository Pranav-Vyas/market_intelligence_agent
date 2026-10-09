"""Web search: Tavily first, `ddgs` as the no-key fallback (task B2)."""

from __future__ import annotations

from mia import fake
from mia.schemas import SearchResult
from mia.stubs import stub


@stub
def search(query: str, k: int = 10) -> list[SearchResult]:
    return fake.search_results(query, k)
