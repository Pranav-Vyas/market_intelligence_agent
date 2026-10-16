from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from mia import runtime
from mia.analysis.clustering import agglomerative, cluster_complaints
from mia.schemas import Complaint


def _c(i: int, summary: str, doc: str | None = None, severity: int = 2, source="hn", company="A"):
    doc = doc or f"d{i}"
    return Complaint(
        id=f"{doc}:0:{i}",
        company=company,
        summary=summary,
        quote=f"quote {i}",
        severity=severity,
        chunk_id=f"{doc}:0",
        doc_id=doc,
        source_type=source,
        url=f"https://example.com/{doc}",
    )


COMPLAINTS = [
    _c(0, "subscription too expensive", source="app_store", company="Otter"),
    _c(1, "subscription too expensive", company="Granola", severity=3),
    _c(2, "subscription price too expensive", source="app_store", company="Otter"),
    _c(3, "calendar sync duplicates events", company="Read"),
    _c(4, "calendar sync duplicates events", source="app_store", company="Otter", severity=1),
    _c(5, "bluetooth headset audio drops"),  # alone: no pain point
]


def _names(req) -> str:
    ids = re.findall(r'<cluster id="(\d+)">\n- ([^\n]+)', req.prompt)
    return json.dumps(
        {"names": [{"cluster_id": int(i), "label": f" {s.upper()}. "} for i, s in ids]}
    )


@pytest.fixture(autouse=True)
def use_llm(settings, llm, backend):
    backend.responder = _names
    runtime.configure(settings=replace(settings, cluster_distance=0.5), llm_client=llm)


def test_similar_summaries_form_clusters_with_stats(backend):
    clusters = cluster_complaints(COMPLAINTS)
    assert [cl.id for cl in clusters] == ["cluster-0", "cluster-1"]
    price, calendar = clusters
    assert price.label == "subscription too expensive"  # normalized from the LLM's label
    assert price.share == pytest.approx(3 / 6)
    assert (price.distinct_docs, price.source_types) == (3, ["app_store", "hn"])
    assert price.companies == ["Granola", "Otter"]
    assert price.mean_severity == pytest.approx(7 / 3, abs=1e-3)
    assert price.complaint_ids[0] == "d1:0:1"  # most severe first
    assert calendar.complaint_ids == ["d3:0:3", "d4:0:4"]
    assert len(backend.calls) == 1  # every cluster is named in one call
    assert backend.calls[0]["model"] == "bulk-model"


def test_complaints_left_alone_form_no_cluster():
    ids = {i for cl in cluster_complaints(COMPLAINTS) for i in cl.complaint_ids}
    assert "d5:0:5" not in ids


def test_at_most_three_complaints_per_document_count():
    thread = [
        _c(i, "subscription too expensive", doc="viral", severity=1 + i % 3) for i in range(6)
    ]
    [cl] = cluster_complaints(thread + [_c(9, "subscription too expensive")])
    assert len(cl.complaint_ids) == 4  # 3 from the viral thread + 1
    assert cl.distinct_docs == 2
    # the most severe three of the thread are kept (severities 3, 3, 2)
    viral = [i for i in cl.complaint_ids if i.startswith("viral")]
    assert viral == ["viral:0:2", "viral:0:5", "viral:0:1"]


def test_unusable_names_fall_back_to_the_most_common_summary(backend):
    backend.responder = None
    backend.responses = ["not json", "still not json"]
    clusters = cluster_complaints(COMPLAINTS)
    assert clusters[0].label == "subscription too expensive"
    assert any(w.startswith("Clustering:") for w in runtime.warnings())


def test_no_complaints_means_no_clusters_and_no_call(backend):
    assert cluster_complaints([]) == []
    assert backend.calls == []


def test_agglomerative_merges_until_the_threshold():
    vectors = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99], [-1, 0]], dtype=float)
    assert agglomerative(vectors, 0.05) == [[0, 1], [2, 3], [4]]
    assert agglomerative(vectors, 0.0001) == [[0], [1], [2], [3], [4]]
    assert agglomerative(vectors, 2.0) == [[0, 1, 2, 3, 4]]
    assert agglomerative(np.zeros((0, 2)), 0.5) == []
