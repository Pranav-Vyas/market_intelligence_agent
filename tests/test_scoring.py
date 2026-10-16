from __future__ import annotations

import pytest

from mia import runtime
from mia.analysis.scoring import pick_evidence, score_opportunities
from mia.schemas import ComplaintCluster, FeatureCell


def _cluster(cid: str, n: int, docs: int, sources: list[str], severity: float, share: float):
    ids = [f"doc{i % docs}:0:{i}" for i in range(n)]
    return ComplaintCluster(
        id=cid,
        label=f"{cid} problem",
        complaint_ids=ids,
        share=share,
        distinct_docs=docs,
        source_types=sources,
        mean_severity=severity,
        companies=["A"],
    )


def _cells(**statuses: str) -> dict[str, FeatureCell]:
    return {n: FeatureCell(company=n, feature="x", status=st) for n, st in statuses.items()}


BIG = _cluster("big", 12, 10, ["app_store", "hn"], 2.5, 0.30)
SMALL = _cluster("small", 6, 4, ["app_store", "hn"], 2.0, 0.15)
ONE_SOURCE = _cluster("one-source", 9, 8, ["app_store"], 3.0, 0.20)
FEW_DOCS = _cluster("few-docs", 6, 2, ["app_store", "hn"], 3.0, 0.15)


@pytest.fixture(autouse=True)
def use_settings(settings):
    runtime.configure(settings=settings)


def test_score_follows_the_formula():
    coverage = {"big": _cells(A="yes", B="partial", C="unknown", D="no")}
    [opp], _ = score_opportunities([BIG], coverage, 4)
    # F = 1, S = 0.75, coverage = 1.5/4 so G = 0.625, E = 1 * (0.5 + 0.5 * 2/3)
    assert opp.score_breakdown == pytest.approx(
        {"F": 1.0, "S": 0.75, "G": 0.625, "E": 0.833}, abs=1e-3
    )
    expected = 10 * (0.35 * 1 + 0.25 * 0.75 + 0.25 * 0.625 + 0.15 * (0.5 + 1 / 3))
    assert opp.score == pytest.approx(expected, abs=0.01)
    assert opp.coverage == 0.375
    assert opp.kind == "unmet_need"  # coverage < 0.4
    assert "1.5/4 competitors claim to address it (yes: A; partly: B)" in opp.rationale


def test_vendors_claiming_it_makes_it_poor_execution():
    coverage = {"big": _cells(A="yes", B="yes", C="unknown", D="no")}
    [opp], _ = score_opportunities([BIG], coverage, 4)
    assert opp.kind == "poor_execution"
    assert opp.score_breakdown["G"] == 0.5


def test_clusters_failing_a_gate_become_weak_signals():
    ranked, weak = score_opportunities([SMALL, ONE_SOURCE, FEW_DOCS, BIG], {}, 4)
    assert [o.cluster_id for o in ranked] == ["big", "small"]  # best first
    assert {o.cluster_id for o in weak} == {"one-source", "few-docs"}
    assert all(o.coverage == 0 and o.kind == "unmet_need" for o in ranked + weak)


def test_frequency_is_relative_to_the_most_common_pain_point():
    ranked, _ = score_opportunities([BIG, SMALL], {}, 4)
    assert {o.cluster_id: o.score_breakdown["F"] for o in ranked} == {"big": 1.0, "small": 0.5}


def test_evidence_spreads_over_documents_first():
    ids = ["d1:0:0", "d1:0:1", "d1:1:2", "d2:0:0", "d3:0:0", "d2:0:1"]
    assert pick_evidence(ids, k=4) == ["d1:0:0", "d2:0:0", "d3:0:0", "d1:0:1"]
    assert pick_evidence(ids) == ["d1:0:0", "d2:0:0", "d3:0:0", "d1:0:1", "d1:1:2", "d2:0:1"]


def test_no_clusters_means_no_opportunities():
    assert score_opportunities([], {}, 5) == ([], [])
