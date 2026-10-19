"""Discovery Scout (task A13): finds the competitors, one search round per graph step.

Round 1 uses the default listicle and comparison queries. While fewer than `min_competitors`
candidates pass homepage validation, the graph loops back here for another round, at most
`discovery.MAX_EXTRA_ROUNDS` extra rounds. For those, the LLM proposes new queries; if it can't,
the scout falls back to "<best candidate so far> alternatives", as the linear pipeline does.
"""

from __future__ import annotations

from mia import discovery, runtime
from mia.agent.actions import NextQueries
from mia.llm import COLLECTION, RECOVERABLE
from mia.schemas import Competitor

MAX_QUERIES = 3

SYSTEM = """You steer the competitor search of a market research agent. Too few products have
passed validation so far. Propose up to 3 new web search queries likely to find articles that
list, review or compare more products in this market, for example "<product> alternatives" or
"<market> for small teams". Never repeat a query that was already tried. "reason" is one short
sentence."""


def step(market: str, progress: dict, round_: int, k: int) -> tuple[dict, list[Competitor], str]:
    """One search round: (new progress, competitors so far, how the queries were chosen)."""
    llm = runtime.llm()
    with llm.phase(COLLECTION):
        if round_ == 0:
            queries, how = discovery.default_queries(market), "default queries"
        else:
            queries, how = next_queries(market, discovery.validated(progress), progress["queries"])
        progress = discovery.run_round(market, progress, queries)
    competitors = discovery.finish(market, progress, k)
    return progress, competitors, f"{how}: {'; '.join(queries)}"


def next_queries(market: str, valid: list[Competitor], tried: list[str]) -> tuple[list[str], str]:
    found = ", ".join(c.name for c in valid) or "none yet"
    prompt = (
        f'Market: "{market}". Products validated so far: {found}.\n'
        "Queries already tried:\n" + "\n".join(f"- {q}" for q in tried) + "\n\nPropose new queries."
    )
    try:
        result = runtime.llm().extract(prompt, schema=NextQueries, system=SYSTEM, model="smart")
    except RECOVERABLE:
        result = NextQueries()
    seen = {q.lower() for q in tried}
    queries = []
    for q in result.queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    if not queries:
        return discovery.alternatives_queries(market, valid), "fallback alternatives queries"
    return queries[:MAX_QUERIES], f"LLM queries ({result.reason or 'no reason given'})"
