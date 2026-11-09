from __future__ import annotations

from typer.testing import CliRunner

from mia.agent import graph
from mia.cli import app
from mia.llm import LLMConfigError, LLMUnavailable


def _stopped_by(monkeypatch, cause: Exception) -> str:
    def stop(*args, **kwargs):
        raise graph.RunStopped("run-1", cause) from cause

    monkeypatch.setattr(graph, "run", stop)
    result = CliRunner().invoke(app, ["run", "AI meeting assistants"])
    assert result.exit_code == 1
    return result.output


def test_a_missing_setting_says_to_fix_env_not_to_resume(monkeypatch):
    out = _stopped_by(monkeypatch, LLMConfigError("MIA_MODEL_BULK is not set."))
    assert "MIA_MODEL_BULK is not set." in out and "Fix .env" in out
    assert "--resume" not in out


def test_an_interrupted_run_says_how_to_resume(monkeypatch):
    out = _stopped_by(monkeypatch, LLMUnavailable("503 high demand"))
    assert "uv run mia run --resume run-1" in out
