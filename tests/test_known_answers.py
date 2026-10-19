"""Clustering and scoring on fixtures where the right answer is known (task B12).

tests/fixtures/complaints.json holds 19 complaints written for three pain points plus two
one-offs. Each complaint's "theme" field is the right answer:
- price (6 complaints, 6 documents, App Store + HN): passes every evidence gate;
- calendar (7 complaints from 2 long threads): capped at 3 per document, only 2 documents;
- consent (4 complaints): too few complaints.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from mia import runtime
from mia.analysis.clustering import cluster_complaints
from mia.analysis.scoring import score_opportunities
from mia.schemas import Complaint, FeatureCell

ROWS = json.loads((Path(__file__).parent / "fixtures" / "complaints.json").read_text())
COMPLAINTS = [Complaint(**{k: v for k, v in r.items() if k != "theme"}) for r in ROWS]
THEME = {r["id"]: r["theme"] for r in ROWS}
COMPETITORS = ["Otter", "Granola", "Fireflies", "Read"]


@pytest.fixture(autouse=True)
def use(settings, llm, backend):
    # name each cluster after the theme of its first complaint, as a careful reader would
    def names(req):
        import re

        out = []
        for cid, first in re.findall(r'<cluster id="(\d+)">\n- ([^\n]+)', req.prompt):
            theme = next(r["theme"] for r in ROWS if r["summary"] == first)
            out.append({"cluster_id": int(cid), "label": theme})
        return json.dumps({"names": out})

    backend.responder = names
    runtime.configure(settings=replace(settings, cluster_distance=0.6), llm_client=llm)


@pytest.fixture
def clusters():
    return {cl.label: cl for cl in cluster_complaints(COMPLAINTS)}


def test_each_cluster_is_exactly_one_theme(clusters):
    assert set(clusters) == {"price", "calendar", "consent"}
    for label, cl in clusters.items():
        assert {THEME[i] for i in cl.complaint_ids} == {label}
    members = defaultdict(set)
    for r in ROWS:
        members[r["theme"]].add(r["id"])
    assert set(clusters["price"].complaint_ids) == members["price"]
    assert set(clusters["consent"].complaint_ids) == members["consent"]


def test_the_per_document_cap_and_the_stats(clusters):
    price, calendar, consent = clusters["price"], clusters["calendar"], clusters["consent"]
    assert (len(price.complaint_ids), price.distinct_docs) == (6, 6)
    assert (len(calendar.complaint_ids), calendar.distinct_docs) == (6, 2)  # 7, capped at 3+3
    assert (len(consent.complaint_ids), consent.distinct_docs) == (4, 4)
    assert price.source_types == ["app_store", "hn"]
    assert price.mean_severity == 2.5
    assert price.share == pytest.approx(6 / 19)
    assert price.companies == ["Fireflies", "Granola", "Otter", "Read"]


def _coverage(clusters, **by_label) -> dict[str, dict[str, FeatureCell]]:
    out = {}
    for label, statuses in by_label.items():
        cl = clusters[label]
        out[cl.id] = {
            name: FeatureCell(company=name, feature=label, status=status)
            for name, status in zip(COMPETITORS, statuses, strict=True)
        }
    return out


def test_only_the_gate_passing_theme_is_ranked(clusters):
    coverage = _coverage(
        clusters,
        price=["unknown"] * 4,
        calendar=["yes", "yes", "partial", "unknown"],
        consent=["no"] * 4,
    )
    ranked, weak = score_opportunities(list(clusters.values()), coverage, len(COMPETITORS))
    assert [o.title for o in ranked] == ["Price"]
    assert {o.title for o in weak} == {"Calendar", "Consent"}
    by_title = {o.title: o for o in ranked + weak}
    assert by_title["Price"].kind == "unmet_need"
    assert by_title["Calendar"].kind == "poor_execution"  # 2.5 of 4 competitors claim it
    assert by_title["Calendar"].coverage == 0.625
    assert by_title["Price"].score_breakdown["F"] == 1.0  # the most common pain point
    assert by_title["Consent"].score_breakdown["S"] == 1.0  # every consent complaint is a 3


def test_ranking_follows_the_score_when_several_pass(clusters, settings):
    runtime.configure(settings=replace(runtime.settings(), gate_min_docs=2, gate_min_complaints=4))
    ranked, weak = score_opportunities(list(clusters.values()), {}, len(COMPETITORS))
    scores = [o.score for o in ranked]
    assert scores == sorted(scores, reverse=True)
    assert {o.title for o in ranked} == {"Price", "Calendar", "Consent"}
    assert weak == []
