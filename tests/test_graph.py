"""The research graph (task A12) against the linear pipeline, on the fixture world."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mia import pipeline
from mia.agent import discovery_scout, graph
from mia.llm import LLM
from tests.conftest import FakeBackend
from tests.test_pipeline import _fake_llm_reply

MARKET = "AI meeting assistants"


@pytest.fixture
def s(settings):
    return replace(settings, max_llm_calls_per_run=50)


def _llm(s, offline: bool = False) -> LLM:
    return LLM(s, offline=offline, backends={"fake": FakeBackend(responder=_fake_llm_reply)})


def _content(report) -> dict:
    data = report.model_dump(mode="json")
    data.pop("generated_at")
    data.pop("stats")
    return data


def test_graph_report_matches_the_linear_pipeline(s, offline_world):
    linear = pipeline.run(MARKET, settings=s, llm_client=_llm(s)).report
    result = graph.run(MARKET, settings=s, llm_client=_llm(s))
    assert _content(result.report) == _content(linear)
    assert result.report.stats["engine"] == "graph"
    assert (result.run_dir / "report.md").exists()


def test_offline_graph_run_replays_the_linear_runs_llm_cache(s, offline_world):
    pipeline.run(MARKET, settings=s, llm_client=_llm(s))
    offline = _llm(s, offline=True)
    result = graph.run(MARKET, offline=True, settings=s, llm_client=offline)
    assert offline.usage.requests == 0
    assert offline.usage.cache_hits > 0
    assert result.report.opportunities  # ranked and verified, not just a skeleton
    assert result.report.stats["warnings"] == []


def test_trace_has_one_entry_per_scout_and_stage(s, offline_world):
    result = graph.run(MARKET, settings=s, llm_client=_llm(s))
    rows = [json.loads(line) for line in (result.run_dir / "trace.jsonl").read_text().splitlines()]
    agents = [r["agent"] for r in rows]
    assert agents.count("product_scout") == 3
    customer = [r for r in rows if r["agent"] == "customer_scout"]
    assert {r["competitor"] for r in customer} == {"Noted", "Scribely", "Recapper"}
    assert {"discovery", "analyst", "verifier", "report"} <= set(agents)
    scouts = [r for r in rows if r["agent"].endswith("_scout")]
    assert len({r["step"] for r in scouts}) == 1  # all six branches ran in the same step
    assert {r["competitor"] for r in scouts} == {"Noted", "Scribely", "Recapper"}
    assert sum(r["requests"] for r in rows) == result.report.stats["llm"]["requests"]
    assert "product_scout" in (result.run_dir / "graph.mmd").read_text()


def test_a_stopped_run_resumes_without_redoing_finished_steps(s, offline_world, monkeypatch):
    calls = {"discover": 0, "official": 0, "analyze": 0}
    discover, official, analyze = (
        discovery_scout.step,
        pipeline.collect_official,
        pipeline.analyze,
    )

    def counting(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    def crash_once(*args, **kwargs):
        calls["analyze"] += 1
        if calls["analyze"] == 1:
            raise RuntimeError("daily quota reached")
        return analyze(*args, **kwargs)

    monkeypatch.setattr(discovery_scout, "step", counting("discover", discover))
    monkeypatch.setattr(pipeline, "collect_official", counting("official", official))
    monkeypatch.setattr(pipeline, "analyze", crash_once)

    with pytest.raises(graph.RunStopped) as stopped:
        graph.run(MARKET, settings=s, llm_client=_llm(s))
    run_id = stopped.value.run_id
    # 3 discovery rounds: the fixture world has 3 competitors, fewer than min_competitors (5)
    assert calls == {"discover": 3, "official": 3, "analyze": 1}

    resumed = _llm(s)
    result = graph.run(resume=run_id, settings=s, llm_client=resumed)
    assert calls == {"discover": 3, "official": 3, "analyze": 2}  # only the failed step reran
    rows = [json.loads(line) for line in (result.run_dir / "trace.jsonl").read_text().splitlines()]
    assert resumed.usage.requests == sum(r["requests"] for r in rows)  # the budget is per run
    assert result.report.opportunities
    assert result.report.stats["run_id"] == run_id


def test_ctrl_c_in_one_scout_branch_reruns_only_that_branch(s, offline_world, monkeypatch):
    calls = {"official": 0, "complaints": 0}
    official, complaints = pipeline.collect_official, pipeline.collect_complaints
    crashed = []

    def count_official(c, choose=None):
        calls["official"] += 1
        return official(c, choose=choose)

    def interrupt_recapper_once(c, market, source, **kwargs):
        calls["complaints"] += 1
        if c.name == "Recapper" and not crashed:
            crashed.append(True)
            raise KeyboardInterrupt
        return complaints(c, market, source, **kwargs)

    monkeypatch.setattr(pipeline, "collect_official", count_official)
    monkeypatch.setattr(pipeline, "collect_complaints", interrupt_recapper_once)
    with pytest.raises(graph.RunStopped) as stopped:
        graph.run(MARKET, settings=s, llm_client=_llm(s))
    before = dict(calls)

    result = graph.run(resume=stopped.value.run_id, settings=s, llm_client=_llm(s))
    assert calls["official"] == before["official"]  # finished branches are not repeated
    assert calls["complaints"] == before["complaints"] + 2  # Recapper's two sources, again
    assert result.report.opportunities


def test_discovery_loops_while_too_few_competitors_validate(s, offline_world, monkeypatch):
    found = offline_world["competitors"]
    extra = [found[0].model_copy(update={"name": f"Extra {i}"}) for i in range(3)]
    rounds = []

    def step(market, progress, round_, k):
        rounds.append(round_)
        return progress, (found + extra)[: 2 + 2 * round_], f"round {round_}"

    monkeypatch.setattr(discovery_scout, "step", step)
    result = graph.run(MARKET, settings=s, llm_client=_llm(s))
    assert rounds == [0, 1, 2]  # 2, then 4 competitors: too few; 6 after the third round
    assert len(result.report.competitors) == 6
    rows = [json.loads(line) for line in (result.run_dir / "trace.jsonl").read_text().splitlines()]
    assert [r["agent"] for r in rows].count("discovery") == 3


def test_discovery_stops_after_two_extra_rounds(s, offline_world, monkeypatch):
    rounds = []

    def step(market, progress, round_, k):
        rounds.append(round_)
        return progress, offline_world["competitors"][:1], f"round {round_}"

    monkeypatch.setattr(discovery_scout, "step", step)
    result = graph.run(MARKET, settings=s, llm_client=_llm(s))
    assert rounds == [0, 1, 2]
    assert [c.name for c in result.report.competitors] == ["Noted"]


def test_resuming_a_finished_run_returns_its_report(s, offline_world):
    first = graph.run(MARKET, settings=s, llm_client=_llm(s))
    again = graph.run(resume=first.report.stats["run_id"], settings=s, llm_client=_llm(s))
    assert again.run_dir == first.run_dir


def test_unknown_run_id_and_missing_market_are_errors(s):
    with pytest.raises(ValueError, match="No saved run"):
        graph.run(resume="no-such-run", settings=s, llm_client=_llm(s))
    with pytest.raises(ValueError, match="Give a market"):
        graph.run(settings=s, llm_client=_llm(s))
