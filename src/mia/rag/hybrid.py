"""HybridIndex: BM25 + dense retrieval fused with RRF, then cross-encoder rerank (tasks C2–C6).

The stub below ranks chunks by simple word overlap so the rest of the pipeline can call the
real interface today.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mia.rag.chunking import chunk_documents
from mia.schemas import Chunk, Document, Evidence, SourceType
from mia.stubs import stub

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class HybridIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    @stub
    def add(self, docs: list[Document]) -> None:
        self.chunks.extend(chunk_documents(docs))

    @stub
    def search(
        self,
        query: str,
        k: int = 10,
        company: str | None = None,
        source_types: list[SourceType] | None = None,
        rerank: bool = True,
    ) -> list[Evidence]:
        q = _words(query)
        scored = []
        for ch in self.chunks:
            if company is not None and ch.company != company:
                continue
            if source_types is not None and ch.source_type not in source_types:
                continue
            overlap = len(q & _words(ch.text))
            if overlap:
                scored.append(Evidence(chunk=ch, score=float(overlap), retriever="stub"))
        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:k]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks])
        )

    @classmethod
    def load(cls, path: Path) -> HybridIndex:
        index = cls()
        index.chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
        return index
