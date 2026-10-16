"""The research graph (task A12): a LangGraph skeleton with one node per pipeline stage.

    discovery ──Send per competitor──► product_scout ──┐
              └──────────────────────► customer_scout ─┴─► analyst ─► verifier ─► report

Product and Customer Scouts run in parallel, one of each per competitor (`max_concurrency` at a
time), and LangGraph joins them before the Analyst. There are no LLM decisions yet: Day 6 adds
the scouts' choices and the Verifier's loop back to targeted research.

Every node calls the same stage functions as `pipeline.run`, and the Analyst puts the scouts'
documents back in the linear order. An offline graph run therefore replays exactly the linear
run's cached LLM calls and writes the same report.

The state is checkpointed to SQLite after every step (`data/runs/checkpoints.sqlite`, keyed by
run id). `mia run --resume <run-id>` continues a stopped run from its last completed step; the
caches make any repeated work free.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from mia import pipeline, runtime, stubs
from mia.agent.state import SCHEMA_TYPES, ResearchState, ScoutTask
from mia.collectors.complaints import COMPLAINT_SOURCES
from mia.collectors.dedupe import dedupe
from mia.config import Settings
from mia.llm import LLM
from mia.schemas import Document, MarketReport, PricingTier

CHECKPOINTS = "checkpoints.sqlite"
PRODUCT_SOURCES = ("official", "pricing", "docs")


class RunStopped(Exception):
    """A graph run stopped before the end; resume it with its run id."""

    def __init__(self, run_id: str, cause: BaseException):
        super().__init__(f"run {run_id} stopped: {cause!r}")
        self.run_id = run_id


# ---- running ------------------------------------------------------------------------------------


def run(
    market: str | None = None,
    *,
    offline: bool = False,
    resume: str | None = None,
    settings: Settings | None = None,
    llm_client: LLM | None = None,
) -> pipeline.RunResult:
    if not (market or resume):
        raise ValueError("Give a market, or a run id to resume.")
    runtime.configure(settings=settings, offline=offline, llm_client=llm_client)
    stubs.reset()
    s = runtime.settings()
    run_id = resume or new_run_id(market or "")
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": s.max_graph_steps,
        "max_concurrency": s.max_concurrency,
    }
    with checkpointer(s) as saver:
        graph = build_graph().compile(checkpointer=saver)
        saved = graph.get_state(config)
        if resume and not saved.values:
            raise ValueError(f"No saved run {resume!r} in {s.runs_dir / CHECKPOINTS}.")
        try:
            if not resume:
                graph.invoke(initial_state(run_id, market or "", offline), config)
            elif "run_dir" not in saved.values:  # the report node is the last step
                # Continue from the last checkpoint. Branches that finished before the stop
                # saved their results as pending writes; only the unfinished ones run again.
                graph.invoke(None, config)
        except (Exception, KeyboardInterrupt) as err:
            raise RunStopped(run_id, err) from err
        final = graph.get_state(config).values
    run_dir = Path(final["run_dir"])
    report = MarketReport.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    return pipeline.RunResult(report=report, run_dir=run_dir)


def new_run_id(market: str) -> str:
    return f"{pipeline.slugify(market)}-{datetime.now():%Y%m%d-%H%M%S}"


def initial_state(run_id: str, market: str, offline: bool) -> ResearchState:
    return {
        "run_id": run_id,
        "market": market,
        "offline": offline,
        "started_at": time.time(),
        "documents": [],
        "pricing": [],
        "complaints": [],
        "trace": [],
        "verify_round": 0,
    }


class checkpointer:  # noqa: N801 (used like a function: `with checkpointer(s) as saver`)
    """A SQLite checkpointer that can only rebuild our own schema classes from a saved state."""

    def __init__(self, s: Settings):
        s.runs_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(s.runs_dir / CHECKPOINTS, check_same_thread=False)

    def __enter__(self) -> SqliteSaver:
        serde = JsonPlusSerializer(allowed_msgpack_modules=list(SCHEMA_TYPES))
        return SqliteSaver(self._conn, serde=serde)

    def __exit__(self, *exc) -> None:
        self._conn.close()


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)
    g.add_node("discovery", discovery)
    g.add_node("product_scout", product_scout)
    g.add_node("customer_scout", customer_scout)
    g.add_node("analyst", analyst)
    g.add_node("verifier", verifier)
    g.add_node("report", report)
    g.add_edge(START, "discovery")
    g.add_conditional_edges("discovery", fan_out, ["product_scout", "customer_scout", "analyst"])
    g.add_edge("product_scout", "analyst")
    g.add_edge("customer_scout", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_edge("verifier", "report")
    g.add_edge("report", END)
    return g


# ---- nodes --------------------------------------------------------------------------------------


def discovery(state: ResearchState, config) -> dict:
    before = runtime.llm().thread_usage()
    competitors = pipeline.discover_competitors(
        state["market"], k=runtime.settings().max_competitors
    )
    summary = f"{len(competitors)} competitors: {', '.join(c.name for c in competitors)}"
    return {
        "competitors": competitors,
        "discovery_round": state.get("discovery_round", 0) + 1,
        "trace": [_trace(state, config, "discovery", "discover", summary, before)],
    }


def fan_out(state: ResearchState) -> list[Send] | str:
    """One Product Scout and one Customer Scout per competitor, all in parallel."""
    tasks = [
        ScoutTask(
            run_id=state["run_id"], market=state["market"], competitor=c, position=i, focus=None
        )
        for i, c in enumerate(state.get("competitors", []))
    ]
    if not tasks:
        return "analyst"
    return [Send("product_scout", t) for t in tasks] + [Send("customer_scout", t) for t in tasks]


def product_scout(task: ScoutTask, config) -> dict:
    before = runtime.llm().thread_usage()
    c = task["competitor"]
    docs = dedupe(pipeline.collect_official(c))
    pricing = pipeline.pricing_for(c, docs)
    pages = sum(d.source_type != "pricing" for d in docs)
    summary = f"{pages} official pages, {len(docs) - pages} pricing pages, {len(pricing)} tiers"
    return {
        "documents": docs,
        "pricing": pricing,
        "trace": [
            _trace(task, config, "product_scout", "collect_official", summary, before, c.name)
        ],
    }


def customer_scout(task: ScoutTask, config) -> dict:
    before = runtime.llm().thread_usage()
    c = task["competitor"]
    limit = runtime.settings().max_complaint_docs_per_source
    docs: list[Document] = []
    counts = []
    for source in COMPLAINT_SOURCES:
        found = pipeline.collect_complaints(c, task["market"], source, limit=limit)
        counts.append(f"{len(found)} {source}")
        docs += found
    return {
        "documents": docs,
        "trace": [
            _trace(task, config, "customer_scout", "collect", ", ".join(counts), before, c.name)
        ],
    }


def analyst(state: ResearchState, config) -> dict:
    before = runtime.llm().thread_usage()
    competitors = state.get("competitors", [])
    docs = linear_order(state.get("documents", []), competitors)
    index = pipeline.build_index(state["market"], docs)
    analysis = pipeline.analyze(competitors, docs, index)
    known = {c.id for c in state.get("complaints", [])}
    summary = (
        f"{len(analysis.complaints)} complaints, {len(analysis.clusters)} clusters, "
        f"{len(analysis.ranked)} ranked, {len(analysis.weak)} weak signals"
    )
    return {
        "features": analysis.features,
        "complaints": [c for c in analysis.complaints if c.id not in known],
        "clusters": analysis.clusters,
        "opportunities": analysis.ranked,
        "weak_signals": analysis.weak,
        "trace": [_trace(state, config, "analyst", "analyze", summary, before)],
    }


def verifier(state: ResearchState, config) -> dict:
    before = runtime.llm().thread_usage()
    _, index = pipeline.load_index(state["market"])
    ranked, verification = pipeline.check(
        state.get("opportunities", []), state.get("complaints", []), index
    )
    summary = (
        f"{verification['opportunities_verified']}/{verification['opportunities']} verified, "
        f"{verification['quotes_passed']}/{verification['quotes_checked']} quotes passed"
    )
    return {
        "opportunities": ranked,
        "verification": verification,
        "verify_round": state.get("verify_round", 0) + 1,
        "trace": [_trace(state, config, "verifier", "verify", summary, before)],
    }


def report(state: ResearchState, config) -> dict:
    before = runtime.llm().thread_usage()
    competitors = state.get("competitors", [])
    docs = linear_order(state.get("documents", []), competitors)
    analysis = pipeline.Analysis(
        features=state.get("features", []),
        complaints=state.get("complaints", []),
        clusters=state.get("clusters", []),
        ranked=state.get("opportunities", []),
        weak=state.get("weak_signals", []),
    )
    result = pipeline.make_report(
        state["market"],
        competitors=competitors,
        docs=docs,
        pricing=_pricing_order(state.get("pricing", []), competitors),
        analysis=analysis,
        ranked=analysis.ranked,
        verification=state.get("verification")
        or pipeline.verification_stats(analysis.ranked, analysis.ranked),
        stats={
            "runtime_s": round(time.time() - state["started_at"], 2),
            "offline": state.get("offline", False),
            "engine": "graph",
            "run_id": state["run_id"],
        },
    )
    run_dir = runtime.settings().runs_dir / pipeline.slugify(state["market"]) / _stamp(state)
    pipeline.write_report(result, run_dir)
    entry = _trace(state, config, "report", "write", str(run_dir), before)
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in [*state.get("trace", []), entry]:
            f.write(json.dumps(row) + "\n")
    diagram = build_graph().compile().get_graph().draw_mermaid()
    (run_dir / "graph.mmd").write_text(diagram, encoding="utf-8")
    return {"run_dir": str(run_dir), "trace": [entry]}


# ---- helpers ------------------------------------------------------------------------------------


def linear_order(docs: list[Document], competitors) -> list[Document]:
    """The scouts' documents in `pipeline.collect` order (per competitor: official pages, then
    each complaint source), without duplicates. Branches finish in any order; this undoes that."""
    rank = {c.name: i for i, c in enumerate(competitors)}

    def key(d: Document) -> tuple[int, int]:
        if d.source_type in PRODUCT_SOURCES:
            kind = 0
        elif d.source_type in COMPLAINT_SOURCES:
            kind = 1 + COMPLAINT_SOURCES.index(d.source_type)
        else:
            kind = len(COMPLAINT_SOURCES) + 1
        return rank.get(d.company or "", len(rank)), kind

    return dedupe(sorted(docs, key=key))  # sorted() is stable: a branch's own order is kept


def _pricing_order(tiers: list[PricingTier], competitors) -> list[PricingTier]:
    rank = {c.name: i for i, c in enumerate(competitors)}
    return sorted(tiers, key=lambda t: rank.get(t.company, len(rank)))


def _stamp(state: ResearchState) -> str:
    match = re.search(r"(\d{8}-\d{6})$", state["run_id"])
    return match.group(1) if match else state["run_id"]


def _trace(
    state, config, agent: str, action: str, summary: str, before: tuple[int, int], competitor=None
) -> dict:
    requests, tokens = runtime.llm().thread_usage()
    return {
        "run_id": state["run_id"],
        "step": (config or {}).get("metadata", {}).get("langgraph_step"),
        "agent": agent,
        "competitor": competitor,
        "action": action,
        "result_summary": summary,
        "requests": requests - before[0],
        "tokens": tokens - before[1],
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
