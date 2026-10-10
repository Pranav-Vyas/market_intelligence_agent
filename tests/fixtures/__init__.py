"""Hand-made Documents for developing and testing without the network (task C3).

The companies (Noted, Scribely, Recapper, ...) are fictional, so we never commit invented
reviews of real products. The set covers official, pricing, reddit, hn, app_store and
comparison sources, including one long page with headings for chunking tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mia.schemas import Document, doc_id

FIXTURE_TIME = datetime(2026, 10, 1, tzinfo=UTC)


def load_documents() -> list[Document]:
    raw = json.loads((Path(__file__).parent / "documents.json").read_text())
    return [Document(id=doc_id(d["url"]), fetched_at=FIXTURE_TIME, **d) for d in raw]
