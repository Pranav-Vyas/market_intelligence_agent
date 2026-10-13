"""End-to-end run of the pipeline on the fixture documents (see `offline_world` in conftest).
Network-bound stages are replaced with fixture data; everything else runs for real."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from mia import pipeline
from mia.llm import LLM
from mia.schemas import MarketReport
from tests.conftest import FakeBackend


def _fake_llm_reply(req) -> str:
    """Complaint extraction: quote the first sentence of every text, so quotes are verbatim."""
    if req.schema is not None and req.schema.__name__ == "_Batch":
        results = []
        texts = re.findall(r'<text id="(\d+)"[^>]*>\n(.*?)\n</text>', req.prompt, re.S)
        for text_id, body in texts:
            text = body.split("\n", 1)[-1].strip()
            first = re.split(r"(?<=[.!?])\s", text)[0]
            summary = "too expensive" if "expensive" in text else "unreliable recording"
            results.append(
                {
                    "text_id": int(text_id),
                    "complaints": [{"summary": summary, "quote": first, "severity": 2}],
                }
            )
        return json.dumps({"results": results})
    if req.schema is not None and req.schema.__name__ == "_Taxonomy":
        names = ["Transcription", "CRM integration", "Calendar integration"]
        return json.dumps({"features": [{"name": n, "description": n} for n in names]})
    if req.schema is not None and req.schema.__name__ == "_Judgments":
        return json.dumps({"judgments": []})
    if req.schema is not None and req.schema.__name__ == "_Pricing":
        return json.dumps({"tiers": [{"name": "Pro", "bucket": "pro", "monthly_usd": 9}]})
    return "{}"


@pytest.fixture
def fake_llm(settings) -> LLM:
    s = replace(settings, max_llm_calls_per_run=50)
    return LLM(s, backends={"fake": FakeBackend(responder=_fake_llm_reply)})


SECTIONS = [
    "## Competitors",
    "## Feature comparison",
    "## Pricing comparison",
    "## Customer complaints",
    "## Potential market gaps",
    "## Sources",
    "## Method and run stats",
]


def test_run_writes_report_and_json(settings, offline_world, fake_llm):
    result = pipeline.run("AI meeting assistants", settings=settings, llm_client=fake_llm)

    report_md = (result.run_dir / "report.md").read_text(encoding="utf-8")
    for section in SECTIONS:
        assert section in report_md

    saved = MarketReport.model_validate(
        json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    )
    assert saved.market == "AI meeting assistants"
    assert result.run_dir.parent.name == "ai-meeting-assistants"


def test_report_is_internally_consistent(settings, offline_world, fake_llm):
    report = pipeline.run("AI meeting assistants", settings=settings, llm_client=fake_llm).report
    names = {c.name for c in report.competitors}
    complaint_ids = {c.id for c in report.complaints}

    assert names == {"Noted", "Scribely", "Recapper"}
    assert report.complaints  # extracted by the real extraction code, with a fake LLM
    assert {c.source_type for c in report.complaints} <= {"hn", "app_store"}
    assert {cell.company for cell in report.features} <= names
    assert {t.company for t in report.pricing} <= names
    for cl in report.clusters:
        assert set(cl.complaint_ids) <= complaint_ids
    for opp in report.opportunities + report.weak_signals:
        assert set(opp.evidence) <= complaint_ids
        assert 0 <= opp.score <= 10
    scores = [o.score for o in report.opportunities]
    assert scores == sorted(scores, reverse=True)


def test_stats_list_stubbed_steps(settings, offline_world, fake_llm):
    report = pipeline.run("AI meeting assistants", settings=settings, llm_client=fake_llm).report
    assert "llm" in report.stats
    assert isinstance(report.stats["stubs"], list)


def test_run_saves_the_index_for_search(settings, offline_world, fake_llm):
    pipeline.run("AI meeting assistants", settings=settings, llm_client=fake_llm)
    slug, index = pipeline.load_index("AI meeting assistants")
    assert slug == "ai-meeting-assistants"
    assert {c.company for c in index.chunks} >= {"Noted", "Scribely", "Recapper"}


def test_slugify():
    assert pipeline.slugify("AI Meeting Assistants!") == "ai-meeting-assistants"
    assert pipeline.slugify("???") == "market"
