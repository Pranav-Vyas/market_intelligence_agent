"""Customer Scout (task B11): collects one competitor's complaints, choosing what to search next.

1. Default searches first: every complaint source with its built-in queries. These are what the
   linear pipeline collects, so they're usually cached.
2. While the competitor has fewer than `min_complaint_docs_per_company` complaint documents,
   the LLM picks the next action from a fixed list, `{"action": "search", "source": "hn",
   "query": "..."}` or `{"action": "done", "reason": "..."}`, and code runs it. A collector
   that fails is reported back as data and never stops the scout.
3. With a `focus` (targeted research asked for by the Verifier), there are no default searches:
   the scout looks only for complaints about that pain point, for `focus_scout_steps` actions.

The scout stops at the threshold, at its step cap, when the LLM says "done", or when collection's
share of the request budget runs out. Its last trace entry says which.
"""

from __future__ import annotations

from mia import pipeline, runtime
from mia.agent.actions import NextAction
from mia.collectors.complaints import COMPLAINT_SOURCES
from mia.collectors.dedupe import dedupe
from mia.llm import COLLECTION, RECOVERABLE
from mia.schemas import Competitor, Document

SOURCE_HELP = {
    "hn": "Hacker News comments; the query is searched next to the product name",
    "app_store": "low-rated App Store reviews; a query keeps the reviews that use its words",
}

SYSTEM = """You steer a research agent that collects public customer complaints about one
software product, for a market report.

Each step, choose one action:
- "search": one source and a query of 1 to 4 words, without the product name (it is added for
  you). Pick words customers use when something goes wrong.
- "done": when more searches are unlikely to find new complaints.

Never repeat a query you already tried on the same source. "reason" is one short sentence."""


def scout(
    c: Competitor,
    market: str,
    *,
    focus: str | None = None,
    known: frozenset[str] = frozenset(),
    tried_before: tuple[str, ...] = (),
) -> tuple[list[Document], list[dict]]:
    """(documents, steps). Each step is a trace row: agent, competitor, action, result_summary,
    requests, tokens; searches also carry source, query, new and focus. `known` holds ids of
    this competitor's documents from earlier rounds, `tried_before` the searches it already ran
    on the same focus, so a second research round tries something new."""
    s = runtime.settings()
    llm = runtime.llm()
    docs: list[Document] = []
    seen = set(known)
    steps: list[dict] = []
    tried: list[str] = [f"{t} (earlier round)" for t in tried_before]
    max_steps = s.focus_scout_steps if focus else s.max_scout_steps

    def run(source: str, query: str | None, label: str, before: tuple[int, int]) -> None:
        try:
            found = pipeline.collect_complaints(
                c, market, source, query=query, limit=s.max_complaint_docs_per_source
            )
            error = ""
        except Exception as err:  # a blocked site or a broken collector is data, not a crash
            found, error = [], f"{type(err).__name__}: {err}"
        new = [d for d in dedupe(found) if d.id not in seen]
        seen.update(d.id for d in new)
        docs.extend(new)
        tried.append(f"{source} {label}: {len(new)} new" + (f" (failed: {error})" if error else ""))
        summary = f"{source} {label}: {len(new)} new documents, {_count(docs, known)} in total"
        row = _step(c, "search", summary + (f"; failed: {error}" if error else ""), before)
        steps.append({**row, "source": source, "query": query, "new": len(new), "focus": focus})

    with llm.phase(COLLECTION):
        if not focus:
            for source in COMPLAINT_SOURCES:
                run(source, None, "default queries", llm.thread_usage())
        while len(steps) < max_steps:
            if not focus and _count(docs, known) >= s.min_complaint_docs_per_company:
                summary = f"reached {_count(docs, known)} complaint documents"
                steps.append(_step(c, "done", summary, llm.thread_usage()))
                return docs, steps
            before = llm.thread_usage()
            try:
                action = llm.extract(
                    _prompt(c, market, focus, tried, max_steps - len(steps), _count(docs, known)),
                    schema=NextAction,
                    system=SYSTEM,
                    model="smart",
                )
            except RECOVERABLE as err:
                steps.append(_step(c, "stop", f"no decision possible ({err})", before))
                return docs, steps
            if action.action == "done":
                steps.append(_step(c, "done", f"scout chose to stop: {action.reason}", before))
                return docs, steps
            query = " ".join(action.query.split())
            label = f'"{query}"'
            if (
                action.source not in COMPLAINT_SOURCES
                or not query
                or any(t.startswith(f"{action.source} {label}:") for t in tried)
            ):
                tried.append(f"{action.source} {label}: invalid or repeated action, skipped")
                steps.append(_step(c, "invalid", f"skipped {action.source} {label}", before))
                continue
            run(action.source, query, label, before)  # the decision counts with its search

    now = llm.thread_usage()
    if focus:
        summary = f"targeted research on '{focus}': {len(docs)} new documents"
        steps.append(_step(c, "done", summary, now))
    else:
        summary = (
            f"step cap reached with {_count(docs, known)} complaint documents "
            f"(target {s.min_complaint_docs_per_company})"
        )
        steps.append(_step(c, "stop", summary, now))
    return docs, steps


def _count(docs: list[Document], known: frozenset[str]) -> int:
    return len(known) + len(docs)


def _prompt(
    c: Competitor, market: str, focus: str | None, tried: list[str], left: int, total: int
) -> str:
    goal = (
        f'Find complaints about this pain point: "{focus}".'
        if focus
        else f"Collect at least {runtime.settings().min_complaint_docs_per_company} complaint "
        f"documents (now {total})."
    )
    sources = "\n".join(f"- {name}: {SOURCE_HELP.get(name, name)}" for name in COMPLAINT_SOURCES)
    history = "\n".join(f"- {t}" for t in tried) or "- nothing yet"
    return (
        f'Product: "{c.name}" ({c.website}), market: "{market}".\n{goal}\n\n'
        f"Sources:\n{sources}\n\nTried so far:\n{history}\n\nSteps left: {left}. "
        "Choose the next action."
    )


def _step(c: Competitor, action: str, summary: str, before: tuple[int, int]) -> dict:
    """A trace row; requests and tokens are those this thread used since `before`."""
    requests, tokens = runtime.llm().thread_usage()
    return {
        "agent": "customer_scout",
        "competitor": c.name,
        "action": action,
        "result_summary": summary,
        "requests": requests - before[0],
        "tokens": tokens - before[1],
    }
