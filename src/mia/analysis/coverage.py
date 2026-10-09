"""Coverage: for each competitor, does its official site address this pain point?
Judged from official docs only, as yes / partial / no with a quote (task B10)."""

from __future__ import annotations

from mia import fake
from mia.rag.hybrid import HybridIndex
from mia.schemas import Competitor, ComplaintCluster, FeatureCell
from mia.stubs import stub


@stub
def assess_coverage(
    cl: ComplaintCluster, comps: list[Competitor], index: HybridIndex
) -> dict[str, FeatureCell]:
    statuses = ("yes", "partial", "no", "no")
    return {
        c.name: FeatureCell(
            company=c.name,
            feature=cl.label,
            status=statuses[fake.pick(c.name + cl.label, len(statuses))],
        )
        for c in comps
    }
