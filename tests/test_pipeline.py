"""End-to-end run of the pipeline. Works with stubs today and keeps working as modules go real
(network-dependent modules must then be exercised offline from fixtures or the cache)."""

from __future__ import annotations

import json

from mia import pipeline
from mia.schemas import MarketReport

SECTIONS = [
    "## Competitors",
    "## Feature comparison",
    "## Pricing comparison",
    "## Customer complaints",
    "## Potential market gaps",
    "## Sources",
    "## Method and run stats",
]


def test_run_writes_report_and_json(settings):
    result = pipeline.run("AI meeting assistants", settings=settings)

    report_md = (result.run_dir / "report.md").read_text()
    for section in SECTIONS:
        assert section in report_md

    saved = MarketReport.model_validate(json.loads((result.run_dir / "run.json").read_text()))
    assert saved.market == "AI meeting assistants"
    assert result.run_dir.parent.name == "ai-meeting-assistants"


def test_report_is_internally_consistent(settings):
    report = pipeline.run("AI meeting assistants", settings=settings).report
    names = {c.name for c in report.competitors}
    complaint_ids = {c.id for c in report.complaints}

    assert len(report.competitors) >= 1
    assert {cell.company for cell in report.features} <= names
    assert {t.company for t in report.pricing} <= names
    for cl in report.clusters:
        assert set(cl.complaint_ids) <= complaint_ids
    for opp in report.opportunities + report.weak_signals:
        assert set(opp.evidence) <= complaint_ids
        assert 0 <= opp.score <= 10
    scores = [o.score for o in report.opportunities]
    assert scores == sorted(scores, reverse=True)


def test_stats_list_stubbed_steps(settings):
    report = pipeline.run("AI meeting assistants", settings=settings).report
    assert "llm" in report.stats
    assert isinstance(report.stats["stubs"], list)


def test_slugify():
    assert pipeline.slugify("AI Meeting Assistants!") == "ai-meeting-assistants"
    assert pipeline.slugify("???") == "market"
