from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from mia import runtime
from mia.analysis.clustering import agglomerative, balanced_share, cluster_complaints
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


def _keep_the_seeds(req) -> str:
    groups = []
    for block in re.findall(r'<group id="\d+">\n(.*?)\n</group>', req.prompt, re.S):
        items = re.findall(r"^(\d+)\. (.+)$", block, re.M)
        groups.append(
            {
                "label": f" {items[0][1].upper()}. ",
                "category": "failure" if "headset" in block else "need",
                "items": [int(i) for i, _ in items],
            }
        )
    return json.dumps({"groups": groups})


def _numbers(req) -> dict[str, int]:
    return {text: int(i) for i, text in re.findall(r"^(\d+)\. (.+)$", req.prompt, re.M)}


@pytest.fixture(autouse=True)
def use_llm(settings, llm, backend):
    backend.responder = _keep_the_seeds
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


def test_the_llm_moves_complaints_between_seeds_and_sets_the_category(backend):
    def regroup(req):
        n = _numbers(req)
        return json.dumps(
            {
                "groups": [
                    {
                        "label": "audio and calendar",
                        "category": "failure",
                        "items": [
                            n["bluetooth headset audio drops"],
                            n["calendar sync duplicates events"],
                            999,  # not a complaint: ignored
                        ],
                    },
                    {
                        "label": "price",
                        "category": "need",
                        # the calendar complaint is already taken: first group wins
                        "items": [
                            n["subscription too expensive"],
                            n["calendar sync duplicates events"],
                        ],
                    },
                ]
            }
        )

    backend.responder = regroup
    audio, price = sorted(cluster_complaints(COMPLAINTS), key=lambda cl: cl.label)
    assert audio.category == "failure" and len(audio.complaint_ids) == 3
    assert price.category == "need"
    assert {i.split(":")[0] for i in price.complaint_ids} == {"d0", "d1"}
    # "subscription price too expensive" wasn't placed, so it stays out of every cluster
    assert all("d2:0:2" not in cl.complaint_ids for cl in (audio, price))


def test_lone_complaints_are_offered_to_the_llm_as_ungrouped(backend):
    cluster_complaints(COMPLAINTS)
    prompt = backend.calls[0]["prompt"]
    assert "bluetooth headset audio drops" in prompt.split("<ungrouped>")[1]


def test_balanced_share_stops_one_source_deciding_what_is_common():
    many = [_c(i, f"review {i}", source="app_store") for i in range(30)]
    few = [_c(100 + i, f"comment {i}", source="hn") for i in range(10)]
    everything = many + few
    assert balanced_share(few[:5], everything) == pytest.approx(0.25)  # half of HN, none of App
    assert balanced_share(many[:5], everything) == pytest.approx(5 / 60)
    # a source with too few complaints can't anchor a share
    assert balanced_share(few[:1], many + few[:2]) == 0.0
    assert balanced_share(few[:1], few[:2]) == 0.5  # unless it is all there is


def test_a_partial_regrouping_is_retried_then_the_seeds_stand(backend):
    def only_one_item(req):
        n = _numbers(req)
        return json.dumps(
            {
                "groups": [
                    {
                        "label": "one group only",
                        "category": "failure",
                        "items": [n["calendar sync duplicates events"]],
                    }
                ]
            }
        )

    # identical summaries count as one item, so this places 2 of the 5 seeded complaints
    backend.responder = only_one_item
    clusters = cluster_complaints(COMPLAINTS)
    assert [c["thinking"] for c in backend.calls] == ["low", "medium"]
    assert {cl.category for cl in clusters} == {"need"}
    assert any("incomplete" in w for w in runtime.warnings())


def test_a_failed_call_counts_every_seed_as_a_need(backend):
    backend.responder = None
    backend.responses = ["not json", "still not json"]
    assert {cl.category for cl in cluster_complaints(COMPLAINTS)} == {"need"}


def test_no_complaints_means_no_clusters_and_no_call(backend):
    assert cluster_complaints([]) == []
    assert backend.calls == []


def test_agglomerative_merges_until_the_threshold():
    vectors = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99], [-1, 0]], dtype=float)
    assert agglomerative(vectors, 0.05) == [[0, 1], [2, 3], [4]]
    assert agglomerative(vectors, 0.0001) == [[0], [1], [2], [3], [4]]
    assert agglomerative(vectors, 2.0) == [[0, 1, 2, 3, 4]]
    assert agglomerative(np.zeros((0, 2)), 0.5) == []
