"""Voice-of-customer collection. Routes to the per-source collectors (tasks B5, B6).

`source` is one of COMPLAINT_SOURCES. The real implementation dispatches to reddit.py, hn.py
and appstores.py, and respects `limit` and `query`.
"""

from __future__ import annotations

from mia import fake
from mia.schemas import Competitor, Document
from mia.stubs import stub

COMPLAINT_SOURCES = ("reddit", "hn", "app_store")


@stub
def collect_complaints(
    c: Competitor,
    market: str,
    source: str,
    query: str | None = None,
    limit: int = 40,
) -> list[Document]:
    if source not in COMPLAINT_SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {COMPLAINT_SOURCES}")
    return [fake.complaint_doc(c.name, source, i) for i in range(min(limit, 4))]
