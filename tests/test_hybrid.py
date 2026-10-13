from __future__ import annotations

import pytest

from mia import runtime
from mia.rag import dense, rerank
from mia.rag.chunking import chunk_documents
from mia.rag.dense import DenseIndex
from mia.rag.hybrid import HybridIndex, rrf
from mia.schemas import Chunk, Evidence
from tests.conftest import FakeEmbedder
from tests.fixtures import load_documents


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return super().embed(texts)


@pytest.fixture
def index() -> HybridIndex:
    index = HybridIndex()
    index.add(load_documents())
    return index


def _hit(chunk_id: str, retriever: str = "bm25") -> Evidence:
    chunk = Chunk(
        id=chunk_id, doc_id=chunk_id, text=chunk_id, company=None, source_type="hn", url="u"
    )
    return Evidence(chunk=chunk, score=1.0, retriever=retriever)


# ---- dense ---------------------------------------------------------------------------------------


def test_dense_finds_the_closest_chunk_and_scores_descend():
    index = DenseIndex()
    index.add(chunk_documents(load_documents()))
    hits = index.search("duplicate calendar invites", k=5)
    assert hits[0].chunk.url == "https://news.example.com/item?id=r3"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert {h.retriever for h in hits} == {"dense"}


def test_dense_filters_by_company_and_source_type():
    index = DenseIndex()
    index.add(chunk_documents(load_documents()))
    hits = index.search("calendar integration", k=20, company="Noted", source_types=["official"])
    assert hits
    assert {(h.chunk.company, h.chunk.source_type) for h in hits} == {("Noted", "official")}
    assert index.search("calendar", source_types=[]) == []
    assert index.search("calendar", company="Nobody") == []
    assert DenseIndex().search("calendar") == []


def test_embeddings_are_cached_on_disk(settings):
    runtime.configure(settings=settings)
    chunks = chunk_documents(load_documents())
    first, second = CountingEmbedder(), CountingEmbedder()
    DenseIndex(first).add(chunks)
    DenseIndex(second).add(chunks)
    assert first.embedded == len(chunks)
    assert second.embedded == 0


def test_adding_the_same_chunks_twice_embeds_them_once():
    embedder = CountingEmbedder()
    index = DenseIndex(embedder)
    chunks = chunk_documents(load_documents())
    index.add(chunks + chunks[:2])  # a batch may repeat a chunk
    index.add(chunks[:5])
    assert embedder.embedded == len(chunks)
    assert len(index.chunks) == len(chunks)


# ---- fusion --------------------------------------------------------------------------------------


def test_rrf_rewards_chunks_that_both_retrievers_return():
    fused = rrf([[_hit("a"), _hit("b"), _hit("c")], [_hit("c", "dense"), _hit("d", "dense")]])
    assert [h.chunk.id for h in fused] == ["c", "a", "b", "d"]  # b and d tie; first seen wins
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert {h.retriever for h in fused} == {"hybrid"}
    assert rrf([[], []]) == []


def test_hybrid_finds_what_bm25_alone_misses(index):
    # no chunk contains these words, so BM25 has nothing; the dense side still returns candidates
    assert index._bm25.search("zebra quantum") == []
    assert index.search("zebra quantum", rerank=False)


def test_hybrid_respects_k_and_filters(index):
    hits = index.search("duplicate calendar invites", k=3, company="Recapper")
    assert 0 < len(hits) <= 3
    assert {h.chunk.company for h in hits} == {"Recapper"}
    hits = index.search("calendar integration", source_types=["official"], rerank=False)
    assert hits
    assert {h.chunk.source_type for h in hits} == {"official"}


def test_rerank_flag_chooses_the_final_scorer(index):
    reranked = index.search("pro plan way too expensive", k=5)
    fused = index.search("pro plan way too expensive", k=5, rerank=False)
    assert {h.retriever for h in reranked} == {"rerank"}
    assert {h.retriever for h in fused} == {"hybrid"}
    assert [h.score for h in reranked] == sorted((h.score for h in reranked), reverse=True)


def test_reranker_reorders_candidates():
    class Backwards:
        def scores(self, query: str, texts: list[str]) -> list[float]:
            return [float(i) for i in range(len(texts))]

    hits = rerank.rerank("q", [_hit("a"), _hit("b"), _hit("c")], Backwards())
    assert [h.chunk.id for h in hits] == ["c", "b", "a"]
    assert rerank.rerank("q", [], Backwards()) == []


def test_save_and_load_keep_results_and_do_not_embed_again(tmp_path, monkeypatch):
    first = CountingEmbedder()
    monkeypatch.setattr(dense, "default_embedder", lambda: first)
    index = HybridIndex()
    index.add(load_documents())
    index.save(tmp_path / "idx")

    second = CountingEmbedder()
    monkeypatch.setattr(dense, "default_embedder", lambda: second)
    monkeypatch.setattr(dense.cache, "get", lambda namespace, key: None)  # no embedding cache
    loaded = HybridIndex.load(tmp_path / "idx")
    assert second.embedded == 0
    assert len(loaded.chunks) == len(index.chunks)
    for query in ("bot joins the call uninvited", "pricing for small teams"):
        assert [e.chunk.id for e in loaded.search(query)] == [
            e.chunk.id for e in index.search(query)
        ]


def test_saving_twice_replaces_the_stored_index(tmp_path):
    docs = load_documents()
    big, small = HybridIndex(), HybridIndex()
    big.add(docs)
    small.add(docs[:3])
    big.save(tmp_path / "idx")
    small.save(tmp_path / "idx")
    loaded = HybridIndex.load(tmp_path / "idx")
    assert len(loaded.chunks) == len(small.chunks)
    assert len(loaded.search("calendar", k=50, rerank=False)) <= len(small.chunks)


@pytest.mark.network
def test_real_models_rank_a_paraphrase_first():
    """Downloads the fastembed models (about 220 MB) on first run."""
    index = HybridIndex()
    index.add(load_documents())
    hits = index.search("costs too much for a small team", k=3)
    assert hits[0].retriever == "rerank"
    assert any("expensive" in h.chunk.text.lower() for h in hits)
