from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from mia.rag.hybrid import HybridIndex
from tests.fixtures import load_documents

_spec = importlib.util.spec_from_file_location(
    "run_retrieval", Path(__file__).parents[1] / "eval" / "run_retrieval.py"
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_recall_mrr_and_ndcg():
    ranked = ["a", "x", "b", "y"]
    relevant = {"a", "b", "c"}
    assert ev.recall(ranked, relevant) == pytest.approx(2 / 3)
    assert ev.mrr(ranked, relevant) == 1.0
    assert ev.mrr(["x", "y", "b"], relevant) == pytest.approx(1 / 3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ev.ndcg(ranked, relevant) == pytest.approx((1 + 1 / math.log2(4)) / ideal)
    assert ev.ndcg(["a", "b", "c"], relevant) == pytest.approx(1.0)


def test_only_the_top_k_documents_count():
    ranked = [f"d{i}" for i in range(20)]
    assert ev.recall(ranked, {"d15"}) == 0.0
    assert ev.mrr(ranked, {"d15"}) == 0.0
    assert ev.recall(ranked, {"d9"}) == 1.0


@pytest.fixture
def index() -> HybridIndex:
    index = HybridIndex()
    index.add(load_documents())
    return index


def test_every_method_is_scored_on_document_level_labels(index):
    calendar = next(d for d in load_documents() if "item?id=r3" in d.url)
    queries = [
        {
            "id": "q1",
            "query": "duplicate calendar invites",
            "relevant": [{"doc_id": calendar.id, "url": calendar.url}],
        }
    ]
    results = ev.evaluate(index, queries)
    assert set(results) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    for r in results.values():
        assert 0 <= r["mean"]["recall@10"] <= 1
        assert r["queries"][0]["id"] == "q1"
    assert ev.as_markdown(results, 1, "test").count("|") > 20


def test_pool_merges_all_methods_and_respects_filters(index):
    queries = [{"id": "q1", "query": "pricing per seat", "company": "Scribely", "relevant": []}]
    rows = ev.pool(index, queries)
    assert rows
    assert {r["company"] for r in rows} == {"Scribely"}
    assert len({r["doc_id"] for r in rows}) == len(rows)
