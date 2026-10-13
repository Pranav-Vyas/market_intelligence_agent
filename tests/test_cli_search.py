from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mia import pipeline, runtime
from mia.cli import app
from tests.fixtures import load_documents

MARKET = "AI meeting assistants"
runner = CliRunner()


@pytest.fixture
def saved(settings):
    runtime.configure(settings=settings)
    return pipeline.build_index(MARKET, load_documents())


def _search(*args: str):
    return runner.invoke(app, ["search", *args])


def test_saved_index_loads_with_the_same_chunks(saved):
    slug, loaded = pipeline.load_index(MARKET)
    assert slug == "ai-meeting-assistants"
    assert [c.id for c in loaded.chunks] == [c.id for c in saved.chunks]


def test_search_filters_by_company_and_source(saved):
    result = _search("calendar duplicate events", "--company", "recapper", "--source", "hn")
    assert result.exit_code == 0, result.output
    assert "Index: ai-meeting-assistants" in result.output
    hits = [line for line in result.output.splitlines() if line[:3].strip().rstrip(".").isdigit()]
    assert hits
    assert all("Recapper · hn ·" in line for line in hits)


def test_company_name_is_matched_loosely(saved):
    result = _search("pricing", "-c", "Noted.ai", "--no-rerank")
    assert result.exit_code == 0, result.output
    assert "Noted ·" in result.output
    assert "Scribely ·" not in result.output


def test_unknown_company_lists_the_indexed_ones(saved):
    result = _search("pricing", "--company", "Otter")
    assert result.exit_code == 1
    assert "Indexed: Noted, Recapper, Scribely" in result.output


def test_unknown_source_is_rejected(saved):
    result = _search("pricing", "--source", "tweets")
    assert result.exit_code == 1
    assert "Unknown source 'tweets'" in result.output


def test_latest_index_is_used_without_a_market(saved, settings):
    pipeline.build_index("Note apps", [d for d in load_documents() if d.company == "Noted"])
    result = _search("pricing")
    assert "Index: note-apps" in result.output
    assert "Index: note-apps" in _search("pricing", "--market", "Note apps").output
    assert "Index: ai-meeting-assistants" in _search("pricing", "-m", MARKET).output


def test_missing_index_explains_what_to_run(settings):
    runtime.configure(settings=settings)
    result = _search("pricing")
    assert result.exit_code == 1
    assert "Run `mia index` first" in result.output


def test_no_hits_says_so(saved):
    result = _search("pricing", "--company", "Noted", "--source", "docs")
    assert result.exit_code == 0
    assert "No matching chunks." in result.output
