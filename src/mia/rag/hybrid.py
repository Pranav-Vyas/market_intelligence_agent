"""HybridIndex (tasks C5, C6): BM25 and dense retrieval fused with reciprocal rank fusion, then
reranked by a cross-encoder.

BM25 finds exact words (product names, "SOC 2"); embeddings find paraphrases ("too expensive"
vs "costs a fortune"). RRF combines the two rankings by rank, so their incomparable scores never
need to be put on one scale. The reranker then reorders the best candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

from mia.rag import dense
from mia.rag import rerank as reranking  # `search` has a parameter named `rerank`
from mia.rag.bm25 import BM25Index
from mia.rag.chunking import chunk_documents
from mia.schemas import Chunk, Document, Evidence, SourceType

RRF_K = 60
CANDIDATES = 50  # taken from each retriever before fusion
RERANK_DEPTH = 30  # fused candidates the cross-encoder reads


class HybridIndex:
    def __init__(
        self,
        *,
        embedder: dense.Embedder | None = None,
        reranker: reranking.Reranker | None = None,
        dense_path: Path | None = None,
    ) -> None:
        self.chunks: list[Chunk] = []
        self._bm25 = BM25Index([])
        self._dense = dense.DenseIndex(embedder, dense_path)
        self._reranker = reranker

    def add(self, docs: list[Document]) -> None:
        self._add_chunks(chunk_documents(docs))

    def _add_chunks(self, chunks: list[Chunk]) -> None:
        known = {c.id for c in self.chunks}
        for chunk in chunks:  # the batch itself may repeat a document
            if chunk.id not in known:
                known.add(chunk.id)
                self.chunks.append(chunk)
        self._bm25 = BM25Index(self.chunks)
        self._dense.add(self.chunks)

    def search(
        self,
        query: str,
        k: int = 10,
        company: str | None = None,
        source_types: list[SourceType] | None = None,
        rerank: bool = True,
    ) -> list[Evidence]:
        depth = max(CANDIDATES, k)
        fused = rrf(
            [
                self._bm25.search(query, k=depth, company=company, source_types=source_types),
                self._dense.search(query, k=depth, company=company, source_types=source_types),
            ]
        )
        if not rerank:
            return fused[:k]
        return reranking.rerank(query, fused[: max(RERANK_DEPTH, k)], self._reranker)[:k]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks]), encoding="utf-8"
        )
        self._dense.save(path / "chroma")

    @classmethod
    def load(cls, path: Path) -> HybridIndex:
        index = cls(dense_path=path / "chroma")
        index._add_chunks(
            [
                Chunk.model_validate(c)
                for c in json.loads((path / "chunks.json").read_text(encoding="utf-8"))
            ]
        )
        return index


def rrf(rankings: list[list[Evidence]], k: int = RRF_K) -> list[Evidence]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to the chunks it returned."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks[hit.chunk.id] = hit.chunk
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [Evidence(chunk=chunks[i], score=scores[i], retriever="hybrid") for i in ordered]
