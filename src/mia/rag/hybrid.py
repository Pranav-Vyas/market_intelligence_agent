"""HybridIndex: BM25 + dense retrieval fused with RRF, then cross-encoder rerank.

Today it is BM25 only (task C2). Dense retrieval and RRF fusion (C4, C5) and the reranker (C6)
plug in behind the same interface, so callers don't change.
"""

from __future__ import annotations

import json
from pathlib import Path

from mia.rag.bm25 import BM25Index
from mia.rag.chunking import chunk_documents
from mia.schemas import Chunk, Document, Evidence, SourceType


class HybridIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25 = BM25Index([])

    def add(self, docs: list[Document]) -> None:
        self._add_chunks(chunk_documents(docs))

    def _add_chunks(self, chunks: list[Chunk]) -> None:
        known = {c.id for c in self.chunks}
        self.chunks.extend(c for c in chunks if c.id not in known)
        self._bm25 = BM25Index(self.chunks)

    def search(
        self,
        query: str,
        k: int = 10,
        company: str | None = None,
        source_types: list[SourceType] | None = None,
        rerank: bool = True,
    ) -> list[Evidence]:
        return self._bm25.search(query, k=k, company=company, source_types=source_types)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks])
        )

    @classmethod
    def load(cls, path: Path) -> HybridIndex:
        index = cls()
        index._add_chunks(
            [Chunk.model_validate(c) for c in json.loads((path / "chunks.json").read_text())]
        )
        return index
