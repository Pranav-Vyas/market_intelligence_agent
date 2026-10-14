from __future__ import annotations

from dataclasses import replace

from typer.testing import CliRunner

from mia import pipeline, runtime
from mia.cli import app
from mia.collectors.stats import collection_stats
from mia.rag.chunking import chunk_documents
from tests.fixtures import load_documents


def _rows():
    return {r.company: r for r in collection_stats(chunk_documents(load_documents()))}


def test_counts_distinct_documents_per_company_and_source():
    noted = _rows()["Noted"]
    expected = {"official": 1, "pricing": 1, "reddit": 3, "hn": 1, "app_store": 1}
    assert noted.docs_by_source == expected
    assert noted.complaint_docs == 5
    assert _rows()["(market)"].docs_by_source == {"comparison": 2}


def test_gaps_name_what_is_missing(settings):
    rows = _rows()
    assert rows["Noted"].gaps(settings) == ["5 complaint docs (minimum 15)"]
    assert rows["Noted"].gaps(replace(settings, min_complaint_docs_per_company=5)) == []
    rows["Noted"].docs_by_source.pop("pricing")
    assert rows["Noted"].gaps(settings)[0] == "no pricing page"


def test_stats_command_reports_each_competitor(settings):
    runtime.configure(settings=replace(settings, min_complaint_docs_per_company=4))
    pipeline.build_index("AI meeting assistants", load_documents())
    result = CliRunner().invoke(app, ["stats"])
    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.strip()}
    assert lines["Noted"].endswith("ok")
    assert lines["Recapper"].endswith("ok")
    assert lines["Scribely"].endswith("3 complaint docs (minimum 4)")
    assert "2/3 competitors meet the minimums." in result.output
