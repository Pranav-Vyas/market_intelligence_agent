"""Smoke test for the Streamlit app (tasks C11, C12). Skipped unless the `ui` extra is
installed: `uv sync --extra ui`."""

from __future__ import annotations

import json
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
