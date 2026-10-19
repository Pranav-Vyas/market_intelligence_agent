"""The Verifier's targets and the research loop back to the Customer Scouts (tasks C12, A14)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mia import runtime
from mia.agent import customer_scout, graph, verifier
from mia.llm import LLM
from mia.schemas import ComplaintCluster, Document, Opportunity, doc_id
from tests.conftest import FakeBackend
from tests.test_pipeline import _fake_llm_reply

MARKET = "AI meeting assistants"


def _doc(company: str, source: str, i: int, text: str) -> Document:
    url = f"https://{source}.example.com/{company.lower()}/{i}"
    return Document(id=doc_id(url), url=url, source_type=source, company=company, text=text)


PRICE = "The plan is too expensive for a small team. We had to drop it."
RECORDING = "It keeps dropping the recording halfway. Lost two calls."


def fake_scout(c, market, *, focus=None, known=frozenset(), tried_before=()):
    """Round one: price complaints only in the App Store, recording complaints only on HN, so
    both are weak signals. Research on the price finds HN comments; on recording, nothing."""
    if focus is None:
        docs = [_doc(c.name, "app_store", i, PRICE) for i in range(2)]
        docs += [_doc(c.name, "hn", i, RECORDING) for i in range(2)]
    elif "expensive" in focus:
        docs = [_doc(c.name, "hn", 10, f"Honestly way too expensive. {PRICE}")]
    else:
        docs = []
    row = {
        "agent": "customer_scout",
        "competitor": c.name,
        "action": "search" if focus else "default",
        "result_summary": f"{len(docs)} documents (focus: {focus})",
        "requests": 0,
        "tokens": 0,
    }
    return docs, [row]


@pytest.fixture
def world(settings, offline_world, monkeypatch):
    monkeypatch.setattr(customer_scout, "scout", fake_scout)
    return replace(settings, max_llm_calls_per_run=80)


def _llm(s) -> LLM:
    return LLM(s, backends={"fake": FakeBackend(responder=_fake_llm_reply)})


def _trace(result) -> list[dict]:
    return [json.loads(line) for line in (result.run_dir / "trace.jsonl").read_text().splitlines()]


def test_research_turns_a_weak_signal_into_a_ranked_opportunity(world):
    result = graph.run(MARKET, settings=world, llm_client=_llm(world))
    report = result.report
    assert [o.title for o in report.opportunities] == ["Too expensive"]
    assert [o.title for o in report.weak_signals] == ["Unreliable recording"]

    rows = _trace(result)
    focus = [r for r in rows if r["agent"] == "customer_scout" and r["action"] == "search"]
    assert {r["competitor"] for r in focus} == {"Noted", "Scribely", "Recapper"}
    compares = [r["result_summary"] for r in rows if r["action"] == "compare"]
    assert any(
        "'Too expensive' weak" in line and "ranked+verified" in line and "CHANGED" in line
        for line in compares
    )
    assert report.stats["agent"]["verify_rounds"] == 3  # 2 research rounds, then the last check
    assert [r["agent"] for r in rows].count("analyst") == 3


def test_research_stops_when_collection_has_no_budget_left(world):
    s = replace(world, scout_budget_share=0.0)
    result = graph.run(MARKET, settings=s, llm_client=_llm(s))
    rows = _trace(result)
    assert [r["agent"] for r in rows].count("analyst") == 1
    targets = next(r for r in rows if r["action"] == "targets")
    assert "collection has used its share of the budget" in targets["result_summary"]


def _opp(title: str, cluster: str, score: float, verified: bool = False) -> Opportunity:
    return Opportunity(
        title=title,
        cluster_id=cluster,
        kind="unmet_need",
        coverage=0,
        score=score,
        score_breakdown={},
        rationale="",
        evidence=[],
        verified=verified,
    )


def _cluster(cid: str, n: int, sources: list[str]) -> ComplaintCluster:
    return ComplaintCluster(
        id=cid,
        label=f"{cid} label",
        complaint_ids=[f"d{i}:0:0" for i in range(n)],
        share=0.1,
        distinct_docs=n,
        source_types=sources,
        mean_severity=2,
        companies=[],
    )


def test_targets_are_unverified_and_weak_opportunities_best_score_first(settings):
    runtime.configure(settings=settings)
    clusters = [
        _cluster("a", 6, ["hn", "app_store"]),
        _cluster("b", 6, ["hn"]),
        _cluster("c", 2, ["hn"]),
    ]
    ranked = [_opp("A", "a", 7.0, verified=True)]
    weak = [_opp("B", "b", 6.0), _opp("C", "c", 6.5)]
    targets = verifier.pick_targets(ranked, weak, clusters, [])
    assert [t["title"] for t in targets] == ["C", "B"]  # A is verified; max 2 targets
    assert targets[0]["why"] == "weak signal: 2/5 complaints, 2/3 documents, 1/2 source types"
    unverified = verifier.pick_targets([_opp("A", "a", 7.0)], weak, clusters, [])
    assert unverified[0]["why"] == "ranked but not verified"


def test_no_research_after_the_last_round(settings):
    runtime.configure(settings=settings, llm_client=_llm(settings))
    allowed, why = verifier.research_allowed({"verify_round": 2}, [{"title": "x"}], [])
    assert not allowed and "2 research rounds done" in why
    s = replace(settings, max_llm_calls_per_run=10)
    runtime.configure(settings=s, llm_client=_llm(s))
    allowed, why = verifier.research_allowed({"verify_round": 0}, [{"title": "x"}], [None] * 20)
    assert not allowed and "another analysis needs" in why
