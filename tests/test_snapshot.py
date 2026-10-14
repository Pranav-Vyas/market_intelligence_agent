from __future__ import annotations

import zipfile
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from mia import cache, pipeline, runtime, snapshot
from mia.cli import app
from mia.llm import LLM
from tests.conftest import FakeBackend
from tests.fixtures import load_documents

MARKET = "AI meeting assistants"


@pytest.fixture
def made(settings, tmp_path):
    """A snapshot made on one machine: an LLM answer, a cached page and a saved index."""
    runtime.configure(settings=settings)
    LLM(settings, backends={"fake": FakeBackend(["cached answer"])}).complete("summarize this")
    cache.put("http", {"url": "https://example.com"}, {"status": 200})
    cache.put("google_play", {"app": "x"}, {"reviews": []})
    pipeline.build_index(MARKET, load_documents())
    (settings.models_dir / "weights.onnx").parent.mkdir(parents=True)
    (settings.models_dir / "weights.onnx").write_bytes(b"model")
    return snapshot.create("test")


@pytest.fixture
def elsewhere(settings, tmp_path):
    """The same settings on a teammate's machine with an empty data directory."""
    s = replace(settings, data_dir=tmp_path / "teammate")
    runtime.configure(settings=s)
    return s


def test_restored_snapshot_answers_llm_calls_and_searches_offline(made, elsewhere):
    manifest = snapshot.restore(made)
    assert manifest["indexes"] == ["ai-meeting-assistants"]
    offline = LLM(elsewhere, offline=True, backends={"fake": FakeBackend()})
    assert offline.complete("summarize this") == "cached answer"
    assert cache.get("http", {"url": "https://example.com"}) == {"status": 200}
    _, index = pipeline.load_index(MARKET)
    assert index.search("calendar", k=3, rerank=False)


def test_models_and_unused_sources_are_left_out_by_default(made, settings):
    names = zipfile.ZipFile(made).namelist()
    assert not any(n.startswith("models/") for n in names)
    assert not any(n.startswith("cache/google_play/") for n in names)
    with_models = snapshot.create("test-models", include_models=True)
    assert "models/weights.onnx" in zipfile.ZipFile(with_models).namelist()


def test_manifest_records_the_settings_an_offline_run_must_match(made, elsewhere):
    manifest = snapshot.restore(made)
    assert manifest["settings"]["MIA_MODEL_BULK"] == "fake/bulk-model"
    assert snapshot.mismatches(manifest, elsewhere) == {}
    other = replace(elsewhere, model_bulk="gemini/other")
    assert snapshot.mismatches(manifest, other) == {
        "MIA_MODEL_BULK": ("fake/bulk-model", "gemini/other")
    }


def test_restore_refuses_paths_outside_the_data_directory(elsewhere, tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr(snapshot.MANIFEST, "{}")
        zf.writestr("../../.bashrc", "echo hi")
    with pytest.raises(snapshot.SnapshotError, match="unexpected path"):
        snapshot.restore(bad)
    assert not (tmp_path / ".bashrc").exists()


def test_restore_rejects_a_zip_without_manifest(elsewhere, tmp_path):
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("cache/x.json", "{}")
    with pytest.raises(snapshot.SnapshotError, match="not a snapshot"):
        snapshot.restore(plain)


def test_empty_data_directory_has_nothing_to_snapshot(elsewhere):
    with pytest.raises(snapshot.SnapshotError, match="Nothing to snapshot"):
        snapshot.create()


def test_restore_command_lists_settings_to_change(made, elsewhere):
    runtime.configure(settings=replace(elsewhere, model_smart="gemini/mine"))
    result = CliRunner().invoke(app, ["snapshot", "restore", str(made)])
    assert result.exit_code == 0, result.output
    assert "MIA_MODEL_SMART=fake/smart-model" in result.output
    assert "MIA_MODEL_BULK" not in result.output
