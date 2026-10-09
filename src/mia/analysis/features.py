"""Feature matrix: LLM proposes 10–15 canonical features from official pages, then judges each
company × feature with a supporting quote (task A7)."""

from __future__ import annotations

from mia import fake
from mia.rag.hybrid import HybridIndex
from mia.schemas import Competitor, FeatureCell
from mia.stubs import stub


@stub
def build_feature_matrix(comps: list[Competitor], index: HybridIndex) -> list[FeatureCell]:
    cells = []
    for c in comps:
        for feature in fake.FEATURES:
            hits = index.search(feature, k=1, company=c.name, source_types=["official"])
            if hits and feature.lower() in hits[0].chunk.text.lower():
                ch = hits[0].chunk
                cells.append(
                    FeatureCell(
                        company=c.name,
                        feature=feature,
                        status="yes",
                        quote=feature,
                        chunk_id=ch.id,
                        url=ch.url,
                    )
                )
            else:
                cells.append(FeatureCell(company=c.name, feature=feature, status="unknown"))
    return cells
