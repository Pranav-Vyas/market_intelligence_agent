"""Smoke test for the Streamlit app (tasks C11, C12, C14). Skipped unless the `ui` extra is
installed: `uv sync --extra ui`."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from mia.agent import graph  # noqa: E402
from mia.config import get_settings  # noqa: E402
from tests.test_graph import _llm  # noqa: E402

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def run_dir(settings, offline_world, monkeypatch):
    s = replace(settings, max_llm_calls_per_run=50)
    result = graph.run("AI meeting assistants", settings=s, llm_client=_llm(s))
    run_json = result.run_dir / "run.json"
    data = json.loads(run_json.read_text())
    data["complaints"][0]["quote"] = "<script>alert(1)</script> " + data["complaints"][0]["quote"]
    run_json.write_text(json.dumps(data))
    monkeypatch.setenv("MIA_DATA_DIR", str(s.data_dir))
    get_settings.cache_clear()
    yield result.run_dir
    get_settings.cache_clear()


def test_app_shows_a_graph_run_without_errors(run_dir):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "AI meeting assistants" in text
    assert "Ranked market gaps" in text and "Agent trace" in text
    assert "<script>alert(1)</script>" not in text  # scraped text is escaped
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    picker = next(sb for sb in at.selectbox if "Select a run" in sb.label)
    assert picker.options[0].startswith("ai-meeting-assistants")


def test_trace_filters_by_competitor(run_dir):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    competitor = next(sb for sb in at.selectbox if sb.label == "Competitor")
    competitor.select("Noted").run()
    assert not at.exception
    assert any("Noted: what each scout did" in m.value for m in at.markdown)


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _evidence_search(at, query: str):
    next(t for t in at.text_input if t.label == "Search the evidence").set_value(query)
    return next(b for b in at.button if b.label == "Search").click().run()


def test_evidence_explorer_searches_the_saved_index(run_dir):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert "Evidence explorer" in _text(at)
    assert "quote-block" not in _text(at).split("Evidence explorer")[1].split("Agent trace")[0]
    at = _evidence_search(at, "price expensive")
    assert not at.exception
    assert any("passages from" in c.value for c in at.caption)
    found = _text(at).split("Evidence explorer")[1].split("Agent trace")[0]
    assert "quote-block" in found and "rerank" in found


def test_evidence_filters_narrow_the_results(run_dir):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(sb for sb in at.selectbox if sb.label == "Company").select("Noted")
    next(c for c in at.checkbox if c.label == "Rerank").uncheck()
    at = _evidence_search(at, "price expensive")
    assert not at.exception
    found = _text(at).split("Evidence explorer")[1].split("Agent trace")[0]
    assert "Noted · " in found and "Scribely · " not in found and "hybrid" in found


def test_evidence_says_so_without_an_index(run_dir, settings):
    shutil.rmtree(settings.data_dir / "indexes")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("No saved index" in m.value for m in at.info)


def test_product_failures_are_shown_apart_from_gaps(run_dir):
    run_json = run_dir / "run.json"
    data = json.loads(run_json.read_text())
    data["opportunities"][0]["kind"] = "product_failure"
    data["opportunities"][0]["title"] = "Crashes on launch"
    run_json.write_text(json.dumps(data))
    text = _text(AppTest.from_file(str(APP), default_timeout=60).run())
    gaps, failures = text.split("Product failures")[0], text.split("### Product failures")[1]
    assert "Crashes on launch" not in gaps
    assert "Crashes on launch" in failures and "product failure" in failures
