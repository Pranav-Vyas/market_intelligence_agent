"""Voice-of-customer collection. Routes to the per-source collectors (tasks B5, B6).

`source` is one of COMPLAINT_SOURCES. `query` narrows the collection to a topic (the Customer
Scout's targeted research), and `limit` caps the documents returned. A source that fails or has
nothing returns an empty list; it never raises.
"""

from __future__ import annotations

from mia.collectors import appstores, hn
from mia.collectors.dedupe import dedupe
from mia.schemas import Competitor, Document

# Reddit is left out: its policy requires approved API access (docs/data_sources.md).
# Add "reddit" here only if that approval comes through.
COMPLAINT_SOURCES = ("hn", "app_store")

_COLLECTORS = {"hn": hn.collect, "app_store": appstores.collect}


def collect_complaints(
    c: Competitor,
    market: str,
    source: str,
    query: str | None = None,
    limit: int = 40,
) -> list[Document]:
    if source not in COMPLAINT_SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {COMPLAINT_SOURCES}")
    return dedupe(_COLLECTORS[source](c, market, query=query, limit=limit))[:limit]
