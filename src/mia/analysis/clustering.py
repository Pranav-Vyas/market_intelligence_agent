"""Cluster complaints: embed summaries → agglomerative clustering → LLM names each cluster →
stats (task B9). The stub groups complaints with identical summaries."""

from __future__ import annotations

from collections import defaultdict

from mia.schemas import Complaint, ComplaintCluster
from mia.stubs import stub


@stub
def cluster_complaints(complaints: list[Complaint]) -> list[ComplaintCluster]:
    groups: dict[str, list[Complaint]] = defaultdict(list)
    for c in complaints:
        groups[c.summary].append(c)
    total = len(complaints) or 1
    clusters = []
    for i, (label, members) in enumerate(sorted(groups.items(), key=lambda g: -len(g[1]))):
        clusters.append(
            ComplaintCluster(
                id=f"cluster-{i}",
                label=label,
                complaint_ids=[m.id for m in members],
                share=len(members) / total,
                distinct_docs=len({m.doc_id for m in members}),
                source_types=sorted({m.source_type for m in members}),
                mean_severity=sum(m.severity for m in members) / len(members),
                companies=sorted({m.company for m in members if m.company}),
            )
        )
    return clusters
