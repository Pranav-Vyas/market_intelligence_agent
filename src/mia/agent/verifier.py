"""Verifier (tasks C12, A14): checks the evidence, then decides what deserves more research.

Each round it:
1. verifies the ranked opportunities (`analysis/verify.py`: verbatim check + LLM judge);
2. compares the opportunities it sent for research last round with what they became, so the
   trace shows whether follow-up research changed them;
3. picks up to `max_focus_targets` weak opportunities for targeted research: ranked ones that
   failed verification, and the best-scoring weak signals (those that missed an evidence gate).
   Each target goes to the Customer Scouts of up to `max_focus_competitors` competitors whose
   customers raised it.

The graph only acts on the targets while research rounds remain (`max_verify_rounds`),
collection still has budget, and enough requests are left for another analysis round.
"""

from __future__ import annotations

from collections import Counter

from mia import pipeline, runtime
from mia.analysis.scoring import passes_gates
from mia.llm import COLLECTION
from mia.schemas import Complaint, ComplaintCluster, Opportunity

ANALYSIS_REQUESTS_PER_CLUSTER = 1  # one coverage call per cluster
ANALYSIS_REQUESTS_FIXED = 6  # cluster names, complaint batches, verification


def verify_round(state: dict) -> tuple[dict, list[dict]]:
    """(state update, trace rows without run metadata)."""
    llm = runtime.llm()
    before = llm.thread_usage()
    _, index = pipeline.load_index(state["market"])
    ranked, verification = pipeline.check(
        state.get("opportunities", []), state.get("complaints", []), index
    )
    rows = [
        _row(
            "verify",
            f"{verification['opportunities_verified']}/{verification['opportunities']} verified, "
            f"{verification['quotes_passed']}/{verification['quotes_checked']} quotes passed",
            before,
        )
    ]
    weak = state.get("weak_signals", [])
    clusters = state.get("clusters", [])
    previous = state.get("research") or []
    rows += [_row("compare", line) for line in compare(previous, ranked, weak, clusters)]
    targets = pick_targets(ranked, weak, clusters, state.get("complaints", []))
    go, why = research_allowed(state, targets, clusters)
    if targets:
        names = "; ".join(
            f"'{t['title']}' ({t['why']}) → {', '.join(t['competitors'])}" for t in targets
        )
        rows.append(_row("targets", f"{names}. {why}"))
    else:
        rows.append(_row("targets", "no weak opportunity to research"))
    update = {
        "opportunities": ranked,
        "verification": verification,
        "verify_round": state.get("verify_round", 0) + 1,
        "research": targets if go else [],
    }
    return update, rows


def pick_targets(
    ranked: list[Opportunity],
    weak: list[Opportunity],
    clusters: list[ComplaintCluster],
    complaints: list[Complaint],
) -> list[dict]:
    s = runtime.settings()
    by_cluster = {cl.id: cl for cl in clusters}
    by_id = {c.id: c for c in complaints}
    candidates = [(o, True, "ranked but not verified") for o in ranked if not o.verified]
    for o in weak:
        cl = by_cluster.get(o.cluster_id)
        if cl is not None:
            candidates.append((o, False, f"weak signal: {missing(cl)}"))
    candidates.sort(key=lambda item: -item[0].score)
    targets = []
    for o, is_ranked, why in candidates[: s.max_focus_targets]:
        cl = by_cluster.get(o.cluster_id)
        if cl is None:
            continue
        owners = [by_id[i].company for i in cl.complaint_ids if i in by_id]
        counts = Counter(name for name in owners if name)
        companies = [name for name, _ in counts.most_common(s.max_focus_competitors)]
        targets.append(
            {
                "title": o.title,
                "label": cl.label,
                "complaint_ids": list(cl.complaint_ids),
                "score": o.score,
                "ranked": is_ranked,
                "verified": o.verified,
                "why": why,
                "competitors": companies,
            }
        )
    return targets


def compare(
    previous: list[dict],
    ranked: list[Opportunity],
    weak: list[Opportunity],
    clusters: list[ComplaintCluster],
) -> list[str]:
    """One line per opportunity researched last round: what it was and what it became. The new
    cluster is the one sharing most of the old cluster's complaints."""
    opps = {o.cluster_id: (o, True) for o in ranked} | {o.cluster_id: (o, False) for o in weak}
    lines = []
    for t in previous:
        old = set(t["complaint_ids"])
        best = max(clusters, key=lambda cl: len(old & set(cl.complaint_ids)), default=None)
        if best is None or not old & set(best.complaint_ids) or best.id not in opps:
            lines.append(f"after research: '{t['title']}' no longer forms a pain point")
            continue
        o, is_ranked = opps[best.id]
        was = _status(t["ranked"], t["verified"])
        now = _status(is_ranked, o.verified)
        added = len(set(best.complaint_ids) - old)
        changed = (was, round(t["score"], 1)) != (now, round(o.score, 1))
        lines.append(
            f"after research: '{t['title']}' {was} {t['score']:.1f} → '{o.title}' {now} "
            f"{o.score:.1f} ({added:+d} complaints){', CHANGED' if changed else ', unchanged'}"
        )
    return lines


def research_allowed(state: dict, targets: list[dict], clusters: list) -> tuple[bool, str]:
    s = runtime.settings()
    llm = runtime.llm()
    if not targets:
        return False, ""
    if state.get("verify_round", 0) >= s.max_verify_rounds:
        return False, f"No more research: {s.max_verify_rounds} research rounds done."
    if not llm.phase_left(COLLECTION):
        return False, "No more research: collection has used its share of the budget."
    left = s.max_llm_calls_per_run - llm.usage.requests
    needed = ANALYSIS_REQUESTS_FIXED + ANALYSIS_REQUESTS_PER_CLUSTER * len(clusters)
    if left < needed:
        return False, f"No more research: {left} requests left, another analysis needs ~{needed}."
    return True, f"Research round {state.get('verify_round', 0) + 1} of {s.max_verify_rounds}."


def missing(cl: ComplaintCluster) -> str:
    """Which evidence gates a cluster misses."""
    s = runtime.settings()
    out = []
    if len(cl.complaint_ids) < s.gate_min_complaints:
        out.append(f"{len(cl.complaint_ids)}/{s.gate_min_complaints} complaints")
    if cl.distinct_docs < s.gate_min_docs:
        out.append(f"{cl.distinct_docs}/{s.gate_min_docs} documents")
    if len(cl.source_types) < s.gate_min_source_types:
        out.append(f"{len(cl.source_types)}/{s.gate_min_source_types} source types")
    return ", ".join(out) or ("passes the gates" if passes_gates(cl) else "unknown")


def _status(is_ranked: bool, verified: bool) -> str:
    if not is_ranked:
        return "weak"
    return "ranked+verified" if verified else "ranked"


def _row(action: str, summary: str, before: tuple[int, int] | None = None) -> dict:
    requests, tokens = runtime.llm().thread_usage()
    before = before or (requests, tokens)
    return {
        "agent": "verifier",
        "competitor": None,
        "action": action,
        "result_summary": summary,
        "requests": requests - before[0],
        "tokens": tokens - before[1],
    }
