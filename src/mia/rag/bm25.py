"""Lexical retrieval with BM25 over chunks (task C2)."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from mia.schemas import Chunk, Evidence, SourceType

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but by can could did do does
    for from had has have how i if in into is it its just me my no not of on or our so some such
    than that the their them then there these they this to too up us was we were what when which
    who will with would you your""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, with a light plural fold (events → event)."""
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    def search(
        self,
        query: str,
        k: int = 10,
        company: str | None = None,
        source_types: list[SourceType] | None = None,
    ) -> list[Evidence]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        hits = [
            (float(score), chunk)
            for score, chunk in zip(scores, self.chunks, strict=True)
            if score > 0
            and (company is None or chunk.company == company)
            and (source_types is None or chunk.source_type in source_types)
        ]
        hits.sort(key=lambda h: h[0], reverse=True)
        return [Evidence(chunk=c, score=s, retriever="bm25") for s, c in hits[:k]]
