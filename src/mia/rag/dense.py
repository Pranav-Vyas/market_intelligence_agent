"""Dense retrieval (task C4): fastembed embeddings in a Chroma collection with metadata filters.

- The embedding model runs locally on CPU through ONNX Runtime (no PyTorch). It is downloaded to
  `data/models` on first use.
- Every embedding is cached on disk by model and text, so re-indexing the same chunks is free.
- The collection lives in memory until `save(path)`; `DenseIndex(path=...)` reopens a saved one
  without embedding anything again.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
import weakref
from functools import cache as memoize
from pathlib import Path
from typing import Protocol

from mia import cache, runtime
from mia.schemas import Chunk, Evidence, SourceType

COLLECTION = "chunks"
EMBED_BATCH = 8
ADD_BATCH = 1000  # Chroma limits how many records one call may add
_COSINE = {"hnsw": {"space": "cosine"}}


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """A fastembed text model, loaded on first use."""

    def __init__(self, model: str):
        self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # A batch is padded to its longest text, so batching similar lengths is about twice as fast
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        model = _load_model(self.name)
        vectors = model.embed([texts[i] for i in order], batch_size=EMBED_BATCH)
        out: list[list[float]] = [[] for _ in texts]
        for i, vector in zip(order, vectors, strict=True):
            out[i] = vector.tolist()
        return out

    def embed_query(self, text: str) -> list[float]:
        return next(iter(_load_model(self.name).query_embed(text))).tolist()


@memoize
def _load_model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=str(runtime.settings().models_dir))


def default_embedder() -> Embedder:  # replaced in tests
    return FastEmbedder(runtime.settings().embed_model)


class DenseIndex:
    def __init__(self, embedder: Embedder | None = None, path: Path | None = None):
        import chromadb
        from chromadb.config import Settings

        self._embedder = embedder or default_embedder()
        self._path = path
        self.chunks: dict[str, Chunk] = {}
        settings = Settings(anonymized_telemetry=False)
        if path is None:
            self._client = chromadb.EphemeralClient(settings=settings)
            name = f"{COLLECTION}-{uuid.uuid4().hex}"
            # in-memory collections are shared by the whole process, so drop ours when done
            weakref.finalize(self, _drop, self._client, name)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=settings)
            name = COLLECTION
        self._collection = self._client.get_or_create_collection(name, configuration=_COSINE)

    def add(self, chunks: list[Chunk]) -> None:
        new = list({c.id: c for c in chunks if c.id not in self.chunks}.values())
        if not new:
            return
        self.chunks.update((c.id, c) for c in new)
        stored = set(self._collection.get(ids=[c.id for c in new], include=[])["ids"])
        todo = [c for c in new if c.id not in stored]
        vectors = self._embed([c.text for c in todo])
        for i in range(0, len(todo), ADD_BATCH):
            batch = todo[i : i + ADD_BATCH]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=vectors[i : i + ADD_BATCH],
                metadatas=[_metadata(c) for c in batch],
            )

    def search(
        self,
        query: str,
        k: int = 10,
        company: str | None = None,
        source_types: list[SourceType] | None = None,
    ) -> list[Evidence]:
        if not self.chunks or not query.strip() or source_types == []:
            return []
        found = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, len(self.chunks)),
            where=_where(company, source_types),
            include=["distances"],
        )
        return [
            Evidence(chunk=self.chunks[chunk_id], score=1.0 - distance, retriever="dense")
            for chunk_id, distance in zip(found["ids"][0], found["distances"][0], strict=True)
            if chunk_id in self.chunks
        ]

    def save(self, path: Path) -> None:
        if self._path is not None and path.resolve() == self._path.resolve():
            return  # already stored there
        target = DenseIndex(self._embedder, path)
        if target._collection.count():
            target._client.delete_collection(COLLECTION)
            target._collection = target._client.create_collection(COLLECTION, configuration=_COSINE)
        ids = list(self.chunks)
        for i in range(0, len(ids), ADD_BATCH):
            got = self._collection.get(ids=ids[i : i + ADD_BATCH], include=["embeddings"])
            target._collection.add(
                ids=got["ids"],
                embeddings=got["embeddings"],
                metadatas=[_metadata(self.chunks[chunk_id]) for chunk_id in got["ids"]],
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [
            {"model": self._embedder.name, "text": hashlib.sha256(t.encode()).hexdigest()}
            for t in texts
        ]
        vectors = [cache.get("embeddings", key) for key in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self._embedder.embed([texts[i] for i in missing])
            for i, vector in zip(missing, fresh, strict=True):
                vectors[i] = [round(x, 6) for x in vector]
                cache.put("embeddings", keys[i], vectors[i])
        return vectors


def _metadata(chunk: Chunk) -> dict:
    # Chroma metadata can't hold None, so market-level chunks get an empty company
    return {"company": chunk.company or "", "source_type": chunk.source_type}


def _where(company: str | None, source_types: list[SourceType] | None) -> dict | None:
    clauses: list[dict] = []
    if company is not None:
        clauses.append({"company": company})
    if source_types is not None:
        clauses.append({"source_type": {"$in": list(source_types)}})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def _drop(client, name: str) -> None:
    with contextlib.suppress(Exception):  # the interpreter may already be shutting down
        client.delete_collection(name)
