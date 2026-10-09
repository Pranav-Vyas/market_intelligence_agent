"""Opportunity scoring and evidence gates (task B10). Formula: SPEC.md §4.

Returns (ranked, weak_signals). Clusters that fail an evidence gate go to weak_signals.
The stub implements the formula and gates; the real version adds the per-document cap and tests.
"""

from __future__ import annotations

from mia import runtime
from mia.schemas import ComplaintCluster, FeatureCell, Opportunity
from mia.stubs import stub


@stub
def score_opportunities(
    clusters: list[ComplaintCluster],
    coverage: dict[str, dict[str, FeatureCell]],  # cluster id -> company -> judgment
    n_competitors: int,
) -> tuple[list[Opportunity], list[Opportunity]]:
    s = runtime.settings()
    w = s.score_weights
    max_share = max((c.share for c in clusters), default=1.0) or 1.0
    ranked, weak = [], []
    for cl in clusters:
        cells = coverage.get(cl.id, {}).values()
        covered = sum(
            1.0 if c.status == "yes" else 0.5 if c.status == "partial" else 0.0 for c in cells
        )
        cov = covered / max(n_competitors, 1)
        parts = {
            "F": cl.share / max_share,
            "S": (cl.mean_severity - 1) / 2,
            "G": 1 - cov,
            "E": min(1.0, cl.distinct_docs / 10) * (0.5 + 0.5 * min(1.0, len(cl.source_types) / 3)),
        }
        score = 10 * sum(w[k] * v for k, v in parts.items())
        opp = Opportunity(
            title=cl.label[:1].upper() + cl.label[1:],
            cluster_id=cl.id,
            kind="unmet_need" if cov < s.unmet_need_coverage_below else "poor_execution",
            coverage=round(cov, 3),
            score=round(score, 2),
            score_breakdown={k: round(v, 3) for k, v in parts.items()},
            rationale=(
                f"{len(cl.complaint_ids)} complaints across {cl.distinct_docs} documents; "
                f"{covered:g}/{n_competitors} competitors address it."
            ),
            evidence=cl.complaint_ids[:5],
        )
        passes = (
            len(cl.complaint_ids) >= s.gate_min_complaints
            and cl.distinct_docs >= s.gate_min_docs
            and len(cl.source_types) >= s.gate_min_source_types
        )
        (ranked if passes else weak).append(opp)
    ranked.sort(key=lambda o: o.score, reverse=True)
    weak.sort(key=lambda o: o.score, reverse=True)
    return ranked, weak
