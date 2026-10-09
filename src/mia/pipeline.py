"""The linear pipeline: every stage, in order, with no agent decisions.

The agent (Day 6) reuses these same stages as tools and decides what to run when. Until a
module's real implementation lands, its stub returns fake data and the report says so.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mia import runtime, stubs
from mia.analysis.clustering import cluster_complaints
from mia.analysis.complaints import extract_complaints
from mia.analysis.coverage import assess_coverage
from mia.analysis.features import build_feature_matrix
from mia.analysis.pricing import extract_pricing
from mia.analysis.scoring import score_opportunities
from mia.analysis.verify import verify
from mia.collectors.complaints import COMPLAINT_SOURCES, collect_complaints
from mia.collectors.web import collect_official
from mia.config import Settings
from mia.discovery import discover_competitors
from mia.rag.chunking import chunk_documents
from mia.rag.hybrid import HybridIndex
from mia.report.markdown import render
from mia.schemas import COMPLAINT_SOURCE_TYPES, Document, MarketReport


@dataclass
class RunResult:
    report: MarketReport
    run_dir: Path


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "market"


def run(market: str, *, offline: bool = False, settings: Settings | None = None) -> RunResult:
    runtime.configure(settings=settings, offline=offline)
    s = runtime.settings()
    stubs.reset()
    started = time.monotonic()

    competitors = discover_competitors(market, k=s.max_competitors)

    docs: list[Document] = []
    for c in competitors:
        docs += collect_official(c)
        for source in COMPLAINT_SOURCES:
            docs += collect_complaints(c, market, source, limit=s.max_complaint_docs_per_source)

    index = HybridIndex()
    index.add(docs)

    features = build_feature_matrix(competitors, index)
    pricing = [
        tier
        for c in competitors
        for tier in extract_pricing(
            c, [d for d in docs if d.company == c.name and d.source_type == "pricing"]
        )
    ]

    complaint_chunks = chunk_documents([d for d in docs if d.source_type in COMPLAINT_SOURCE_TYPES])
    complaints = extract_complaints(complaint_chunks)
    clusters = cluster_complaints(complaints)
    coverage = {cl.id: assess_coverage(cl, competitors, index) for cl in clusters}
    ranked, weak = score_opportunities(clusters, coverage, len(competitors))
    ranked = verify(ranked, complaints, index)
    verification_rate = sum(o.verified for o in ranked) / len(ranked) if ranked else None

    report = MarketReport(
        market=market,
        competitors=competitors,
        features=features,
        pricing=pricing,
        complaints=complaints,
        clusters=clusters,
        opportunities=ranked,
        weak_signals=weak,
        stats={
            "documents_by_source": dict(Counter(d.source_type for d in docs)),
            "llm": runtime.llm().usage.as_dict(),
            "verification_rate": verification_rate,
            "runtime_s": round(time.monotonic() - started, 2),
            "offline": offline,
            "stubs": stubs.used(),
        },
    )

    run_dir = s.runs_dir / slugify(market) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(render(report))
    (run_dir / "run.json").write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return RunResult(report=report, run_dir=run_dir)
