from __future__ import annotations

import pytest

from mia import runtime
from mia.analysis.scoring import pick_evidence, score_opportunities
from mia.schemas import ComplaintCluster, FeatureCell


def _cluster(
    cid: str,
    n: int,
    docs: int,
    sources: list[str],
    severity: float,
    share: float,
    *,
    category: str = "need",
    balanced: float | None = None,
    companies: tuple[str, ...] = ("A", "B"),
):
    ids = [f"doc{i % docs}:0:{i}" for i in range(n)]
    return ComplaintCluster(
        id=cid,
        label=f"{cid} problem",
        complaint_ids=ids,
        share=share,
        distinct_docs=docs,
        source_types=sources,
        mean_severity=severity,
        companies=list(companies),
        category=category,
        balanced_share=balanced,
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
    assert opp.rationale == "Claimed on the sites of A, B (partly)."


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
    assert ranked[0].rationale == "No competitor's own site claims to address it."


def test_frequency_is_relative_to_the_most_common_pain_point():
    ranked, _ = score_opportunities([BIG, SMALL], {}, 4)
    assert {o.cluster_id: o.score_breakdown["F"] for o in ranked} == {"big": 1.0, "small": 0.5}


def test_frequency_uses_the_source_balanced_share_when_there_is_one():
    hn_heavy = _cluster("hn-heavy", 6, 6, ["app_store", "hn"], 2.0, 0.10, balanced=0.30)
    app_heavy = _cluster("app-heavy", 12, 10, ["app_store", "hn"], 2.0, 0.30, balanced=0.15)
    ranked, _ = score_opportunities([hn_heavy, app_heavy], {}, 4)
    assert {o.cluster_id: o.score_breakdown["F"] for o in ranked} == {
        "hn-heavy": 1.0,
        "app-heavy": 0.5,
    }


def test_a_product_failure_is_scored_on_its_spread_not_on_vendor_claims():
    crashes = _cluster(
        "crashes",
        12,
        10,
        ["app_store"],
        3.0,
        0.30,
        category="failure",
        companies=("A", "B", "C"),
    )
    # a stray coverage judgment must not count: failures are never compared with vendor claims
    [opp], weak = score_opportunities([crashes], {"crashes": _cells(A="yes", B="yes")}, 4)
    assert weak == []
    assert opp.kind == "product_failure"
    assert opp.coverage == 0
    assert opp.score_breakdown["G"] == 0.75  # 3 of 4 competitors
    assert opp.rationale.startswith("Customers of 3 of 4 competitors report it.")


def test_gaps_rank_before_failures_whatever_the_score():
    crashes = _cluster("crashes", 12, 10, ["app_store"], 3.0, 0.3, category="failure")
    ranked, _ = score_opportunities([crashes, SMALL], {}, 4)
    assert [o.cluster_id for o in ranked] == ["small", "crashes"]
    assert ranked[1].score > ranked[0].score


def test_frequency_is_relative_within_each_category():
    crashes = _cluster("crashes", 12, 10, ["app_store"], 3.0, 0.30, category="failure")
    ranked, _ = score_opportunities([crashes, SMALL], {}, 4)
    assert {o.cluster_id: o.score_breakdown["F"] for o in ranked} == {"crashes": 1.0, "small": 1.0}


def test_one_vendors_problem_is_not_a_market_gap():
    only_a = _cluster("only-a", 12, 10, ["app_store", "hn"], 3.0, 0.3, companies=("A",))
    ranked, weak = score_opportunities([only_a], {}, 4)
    assert ranked == []
    assert [o.cluster_id for o in weak] == ["only-a"]


def test_a_failure_needs_no_second_source_but_does_need_a_second_company():
    both = _cluster("both", 12, 10, ["app_store"], 3.0, 0.3, category="failure")
    one = _cluster("one", 12, 10, ["app_store"], 3.0, 0.3, category="failure", companies=("A",))
    ranked, weak = score_opportunities([both, one], {}, 4)
    assert [o.cluster_id for o in ranked] == ["both"]
    assert [o.cluster_id for o in weak] == ["one"]


def test_evidence_spreads_over_documents_first():
    ids = ["d1:0:0", "d1:0:1", "d1:1:2", "d2:0:0", "d3:0:0", "d2:0:1"]
    assert pick_evidence(ids, k=4) == ["d1:0:0", "d2:0:0", "d3:0:0", "d1:0:1"]
    assert pick_evidence(ids) == ["d1:0:0", "d2:0:0", "d3:0:0", "d1:0:1", "d1:1:2", "d2:0:1"]


def test_no_clusters_means_no_opportunities():
    assert score_opportunities([], {}, 5) == ([], [])
