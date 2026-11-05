"""Opportunity scoring and evidence gates (tasks B10, B13). Formula and gates: SPEC.md §4.

    F = balanced_share / max_balanced_share    relative frequency, within the cluster's category
    S = (mean_severity - 1) / 2                severity, 0..1
    G = 1 - coverage                           competitor gap
    E = min(1, docs/10) · (0.5 + 0.5·min(1, source_types/3))   evidence strength
    score = 10 · (wF·F + wS·S + wG·G + wE·E)   weights in config.py

A product failure (crashes, lost data, charges after cancelling) is not a market gap: no vendor
advertises "we don't crash", so "nobody claims to solve it" says nothing. Failures skip coverage,
use the share of competitors whose customers report them as G, get the kind "product_failure"
and rank after every real gap.

Scores are plain code, never an LLM's opinion. Clusters already count at most
`max_complaints_per_doc` complaints per document (clustering.py). A cluster that fails a gate
(too few complaints, documents, companies or source types) becomes a weak signal.

Returns (ranked, weak_signals), each with gaps first, best first.
"""

from __future__ import annotations

from mia import runtime
from mia.analysis.coverage import coverage_share
from mia.schemas import ComplaintCluster, FeatureCell, Opportunity

EVIDENCE_CANDIDATES = 8  # complaint ids handed to verify.py, which keeps those that pass


def score_opportunities(
    clusters: list[ComplaintCluster],
    coverage: dict[str, dict[str, FeatureCell]],  # cluster id -> company -> judgment
    n_competitors: int,
) -> tuple[list[Opportunity], list[Opportunity]]:
    s = runtime.settings()
    max_share = {
        category: max((frequency(c) for c in clusters if c.category == category), default=1.0)
        or 1.0
        for category in ("failure", "need")
    }
    ranked, weak = [], []
    for cl in clusters:
        cells = coverage.get(cl.id, {})
        failure = cl.category == "failure"
        cov = 0.0 if failure else coverage_share(cells, n_competitors)
        gap = len(cl.companies) / max(n_competitors, 1) if failure else 1 - cov
        parts = breakdown(cl, gap, max_share[cl.category])
        if failure:
            kind = "product_failure"
        else:
            kind = "unmet_need" if cov < s.unmet_need_coverage_below else "poor_execution"
        opp = Opportunity(
            title=cl.label[:1].upper() + cl.label[1:],
            cluster_id=cl.id,
            kind=kind,
            coverage=round(cov, 3),
            score=round(10 * sum(s.score_weights[k] * v for k, v in parts.items()), 2),
            score_breakdown={k: round(v, 3) for k, v in parts.items()},
            rationale=_failure_rationale(cl, n_competitors) if failure else _rationale(cells),
            evidence=pick_evidence(cl.complaint_ids),
        )
        (ranked if passes_gates(cl) else weak).append(opp)
    ranked.sort(key=_rank)
    weak.sort(key=_rank)
    return ranked, weak


def _rank(o: Opportunity) -> tuple[bool, float]:
    return o.kind == "product_failure", -o.score


def frequency(cl: ComplaintCluster) -> float:
    return cl.share if cl.balanced_share is None else cl.balanced_share


def breakdown(cl: ComplaintCluster, gap: float, max_share: float) -> dict[str, float]:
    return {
        "F": frequency(cl) / max_share,
        "S": (cl.mean_severity - 1) / 2,
        "G": gap,
        "E": min(1.0, cl.distinct_docs / 10) * (0.5 + 0.5 * min(1.0, len(cl.source_types) / 3)),
    }


def passes_gates(cl: ComplaintCluster) -> bool:
    s = runtime.settings()
    enough_sources = cl.category == "failure" or len(cl.source_types) >= s.gate_min_source_types
    return (
        len(cl.complaint_ids) >= s.gate_min_complaints
        and cl.distinct_docs >= s.gate_min_docs
        and len(cl.companies) >= s.gate_min_companies
        and enough_sources
    )


def pick_evidence(complaint_ids: list[str], k: int = EVIDENCE_CANDIDATES) -> list[str]:
    """Up to k complaints, at most one per document until every document has had a turn.
    The ids come most severe first, and a complaint id starts with its document id."""
    first, rest, seen = [], [], set()
    for cid in complaint_ids:
        doc = cid.split(":", 1)[0]
        (rest if doc in seen else first).append(cid)
        seen.add(doc)
    return (first + rest)[:k]


def _failure_rationale(cl: ComplaintCluster, n_competitors: int) -> str:
    return (
        f"Customers of {len(cl.companies)} of {n_competitors} competitors report it. A vendor "
        "wouldn't advertise against this, so its site wasn't checked."
    )


def _rationale(cells: dict[str, FeatureCell]) -> str:
    """Which competitors' own sites claim to address the pain point."""
    claims = [name for name, c in cells.items() if c.status == "yes"]
    claims += [f"{name} (partly)" for name, c in cells.items() if c.status == "partial"]
    if not claims:
        return "No competitor's own site claims to address it."
    return f"Claimed on the sites of {', '.join(claims)}."
