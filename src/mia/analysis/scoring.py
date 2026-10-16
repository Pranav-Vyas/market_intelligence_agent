"""Opportunity scoring and evidence gates (task B10). Formula and gates: SPEC.md §4.

    F = share / max_share                      relative frequency
    S = (mean_severity - 1) / 2                severity, 0..1
    G = 1 - coverage                           competitor gap
    E = min(1, docs/10) · (0.5 + 0.5·min(1, source_types/3))   evidence strength
    score = 10 · (wF·F + wS·S + wG·G + wE·E)   weights in config.py

Scores are plain code, never an LLM's opinion. Clusters already count at most
`max_complaints_per_doc` complaints per document (clustering.py). A cluster that fails a gate
(too few complaints, documents or source types) becomes a weak signal instead of being ranked.

Returns (ranked, weak_signals), each best first.
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
    max_share = max((c.share for c in clusters), default=1.0) or 1.0
    ranked, weak = [], []
    for cl in clusters:
        cells = coverage.get(cl.id, {})
        cov = coverage_share(cells, n_competitors)
        parts = breakdown(cl, cov, max_share)
        opp = Opportunity(
            title=cl.label[:1].upper() + cl.label[1:],
            cluster_id=cl.id,
            kind="unmet_need" if cov < s.unmet_need_coverage_below else "poor_execution",
            coverage=round(cov, 3),
            score=round(10 * sum(s.score_weights[k] * v for k, v in parts.items()), 2),
            score_breakdown={k: round(v, 3) for k, v in parts.items()},
            rationale=_rationale(cells),
            evidence=pick_evidence(cl.complaint_ids),
        )
        (ranked if passes_gates(cl) else weak).append(opp)
    ranked.sort(key=lambda o: o.score, reverse=True)
    weak.sort(key=lambda o: o.score, reverse=True)
    return ranked, weak


def breakdown(cl: ComplaintCluster, coverage: float, max_share: float) -> dict[str, float]:
    return {
        "F": cl.share / max_share,
        "S": (cl.mean_severity - 1) / 2,
        "G": 1 - coverage,
        "E": min(1.0, cl.distinct_docs / 10) * (0.5 + 0.5 * min(1.0, len(cl.source_types) / 3)),
    }


def passes_gates(cl: ComplaintCluster) -> bool:
    s = runtime.settings()
    return (
        len(cl.complaint_ids) >= s.gate_min_complaints
        and cl.distinct_docs >= s.gate_min_docs
        and len(cl.source_types) >= s.gate_min_source_types
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


def _rationale(cells: dict[str, FeatureCell]) -> str:
    """Which competitors' own sites claim to address the pain point."""
    claims = [name for name, c in cells.items() if c.status == "yes"]
    claims += [f"{name} (partly)" for name, c in cells.items() if c.status == "partial"]
    if not claims:
        return "No competitor's own site claims to address it."
    return f"Claimed on the sites of {', '.join(claims)}."
