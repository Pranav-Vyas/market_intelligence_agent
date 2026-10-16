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
from mia.analysis.verify import verification_stats, verify
from mia.collectors.complaints import COMPLAINT_SOURCES, collect_complaints
from mia.collectors.dedupe import dedupe
from mia.collectors.web import collect_official
from mia.config import Settings
from mia.discovery import discover_competitors
from mia.llm import LLM
from mia.rag.chunking import chunk_documents
from mia.rag.hybrid import HybridIndex
from mia.report.markdown import render
from mia.schemas import COMPLAINT_SOURCE_TYPES, Competitor, Document, MarketReport


@dataclass
class RunResult:
    report: MarketReport
    run_dir: Path


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "market"


def run(
    market: str,
    *,
    offline: bool = False,
    settings: Settings | None = None,
    llm_client: LLM | None = None,
) -> RunResult:
    runtime.configure(settings=settings, offline=offline, llm_client=llm_client)
    s = runtime.settings()
    stubs.reset()
    started = time.monotonic()

    competitors = discover_competitors(market, k=s.max_competitors)
    docs = collect(market, competitors)
    index = build_index(market, docs)

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
    checked = verify(ranked, complaints, index)
    verification = verification_stats(ranked, checked)
    ranked = checked

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
            "verification_rate": verification["verification_rate"],
            "verification": verification,
            "runtime_s": round(time.monotonic() - started, 2),
            "offline": offline,
            "stubs": stubs.used(),
            "warnings": runtime.warnings(),
        },
    )

    run_dir = s.runs_dir / slugify(market) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(render(report), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return RunResult(report=report, run_dir=run_dir)


def collect(market: str, competitors: list[Competitor]) -> list[Document]:
    """Official pages and complaints for every competitor."""
    s = runtime.settings()
    docs: list[Document] = []
    for c in competitors:
        docs += collect_official(c)
        for source in COMPLAINT_SOURCES:
            docs += collect_complaints(c, market, source, limit=s.max_complaint_docs_per_source)
    return dedupe(docs)  # per company: a comment naming two competitors stays for both


def index_dir(market: str) -> Path:
    return runtime.settings().indexes_dir / slugify(market)


def build_index(market: str, docs: list[Document]) -> HybridIndex:
    """Index the documents and save the index for `mia search`, replacing the market's last one."""
    index = HybridIndex()
    index.add(docs)
    index.save(index_dir(market))
    return index


def load_index(market: str | None = None) -> tuple[str, HybridIndex]:
    """(market slug, index) of a saved index: the market's, or the most recently saved one."""
    root = runtime.settings().indexes_dir
    if market:
        path: Path | None = index_dir(market)
    else:
        saved = sorted(root.glob("*/chunks.json"), key=lambda p: p.stat().st_mtime)
        path = saved[-1].parent if saved else None
    if path is None or not (path / "chunks.json").exists():
        target = f'"{market}"' if market else "any market"
        raise FileNotFoundError(f"No saved index for {target} in {root}. Run `mia index` first.")
    return path.name, HybridIndex.load(path)
