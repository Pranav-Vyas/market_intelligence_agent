from __future__ import annotations

import pytest

from mia.rag.bm25 import BM25Index, tokenize
from mia.rag.chunking import chunk_documents
from mia.rag.hybrid import HybridIndex
from tests.fixtures import load_documents


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index(chunk_documents(load_documents()))


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The integrations are way too expensive!") == [
        "integration",
        "way",
        "expensive",
    ]
    assert tokenize("access process") == ["access", "process"]


def test_finds_price_complaints(index):
    # BM25 needs shared words. "too expensive for a small team" misses the post about a "tiny
    # team"; that synonym gap is what dense retrieval (C4) is for, and belongs in the eval set.
    urls = [e.chunk.url for e in index.search("pro plan way too expensive", k=3)]
    assert "https://reddit.example.com/r/productivity/comments/n1" in urls
    assert "https://reddit.example.com/r/sales/comments/r1" in urls


def test_scores_are_descending_and_labelled(index):
    hits = index.search("calendar duplicate events", k=5)
    assert hits
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert {h.retriever for h in hits} == {"bm25"}


def test_company_filter(index):
    hits = index.search("duplicate calendar invites", company="Recapper")
    assert hits
    assert {h.chunk.company for h in hits} == {"Recapper"}
    assert hits[0].chunk.url == "https://news.example.com/item?id=r3"


def test_source_type_filter(index):
    hits = index.search("calendar integration", source_types=["official"])
    assert hits
    assert {h.chunk.source_type for h in hits} == {"official"}


def test_no_match_and_stopword_only_queries_return_nothing(index):
    assert index.search("zebra quantum") == []
    assert index.search("the and of") == []


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything") == []


def test_hybrid_index_round_trips_and_ignores_duplicate_docs(tmp_path):
    docs = load_documents()
    index = HybridIndex()
    index.add(docs)
    index.add(docs[:3])  # re-adding the same documents must not duplicate chunks
    n = len(index.chunks)
    index.save(tmp_path / "idx")

    loaded = HybridIndex.load(tmp_path / "idx")
    assert len(loaded.chunks) == n
    query = "bot joins the call uninvited"
    assert [e.chunk.id for e in loaded.search(query)] == [e.chunk.id for e in index.search(query)]
