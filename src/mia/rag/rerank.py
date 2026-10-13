"""Cross-encoder reranking (task C6) over the fused candidates.

A cross-encoder reads the query and a chunk together, which is slower but more accurate than
comparing embeddings. It runs locally on CPU through fastembed (ONNX Runtime) and is downloaded
to `data/models` on first use.
"""

from __future__ import annotations

from functools import cache as memoize
from typing import Protocol

from mia import runtime
from mia.schemas import Evidence

# Time grows with text length, and the start of a chunk (its header and first sentences) says
# what it is about. Without this limit, 30 long forum comments take about 8 s on a laptop CPU.
MAX_CHARS = 1000


class Reranker(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A fastembed cross-encoder, loaded on first use."""

    def __init__(self, model: str):
        self.name = model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in _load_model(self.name).rerank(query, texts)]


@memoize
def _load_model(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(name, cache_dir=str(runtime.settings().models_dir))


def default_reranker() -> Reranker:  # replaced in tests
    return CrossEncoderReranker(runtime.settings().rerank_model)


def rerank(query: str, hits: list[Evidence], reranker: Reranker | None = None) -> list[Evidence]:
    """The same chunks, best first, scored by the cross-encoder."""
    if not hits:
        return []
    texts = [h.chunk.text[:MAX_CHARS] for h in hits]
    scores = (reranker or default_reranker()).scores(query, texts)
    ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: pair[0], reverse=True)
    return [Evidence(chunk=h.chunk, score=s, retriever="rerank") for s, h in ranked]
