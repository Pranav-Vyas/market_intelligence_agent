"""Costs and multi-agent vs linear (task A15, SPEC.md §10).

    uv run python eval/compare_runs.py <graph run.json> <linear run.json> \
        [--fresh <run.json>] [--replay <run.json>]    # → eval/results/graph_vs_linear.{json,md}

"Calls" counts requests plus cache hits: what a run costs from an empty cache. Wall-clock time and
tokens are only meaningful for the run that really made the calls (--fresh).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    stats, llm = report["stats"], report["stats"].get("llm", {})
    opportunities = report["opportunities"]
    verification = stats.get("verification") or {}
    return {
        "run": path.parent.name,
        "engine": stats.get("engine", "linear"),
        "competitors": len(report["competitors"]),
        "documents_by_source": stats.get("documents_by_source", {}),
        "complaint_documents_by_company": stats.get("complaint_documents_by_company", {}),
        "complaints": len(report["complaints"]),
        "clusters": len(report["clusters"]),
        "market_gaps": sum(o["kind"] != "product_failure" for o in opportunities),
        "product_failures": sum(o["kind"] == "product_failure" for o in opportunities),
        "verified_market_gaps": sum(
            o["verified"] and o["kind"] != "product_failure" for o in opportunities
        ),
        "weak_signals": len(report["weak_signals"]),
        "quote_pass_rate": verification.get("quote_pass_rate"),
        "requests": llm.get("requests", 0),
        "cache_hits": llm.get("cache_hits", 0),
        "calls": llm.get("requests", 0) + llm.get("cache_hits", 0),
        "input_tokens": llm.get("input_tokens", 0),
        "output_tokens": llm.get("output_tokens", 0),
        "fallbacks": llm.get("fallbacks", 0),
        "requests_by_model": llm.get("requests_by_model", {}),
        "runtime_s": stats.get("runtime_s"),
        "offline": stats.get("offline", False),
        "top_gaps": [o["title"] for o in opportunities if o["kind"] != "product_failure"][:5],
    }


ROWS = [
    ("Competitors", "competitors"),
    ("Complaints", "complaints"),
    ("Pain points", "clusters"),
    ("Ranked market gaps", "market_gaps"),
    ("Verified market gaps", "verified_market_gaps"),
    ("Product failures", "product_failures"),
    ("Weak signals", "weak_signals"),
    ("Quote pass rate", "quote_pass_rate"),
    ("LLM calls from an empty cache", "calls"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("linear", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    graph, linear = summarize(args.graph), summarize(args.linear)
    out = {"graph": graph, "linear": linear}
    lines = [
        "# Multi-agent graph vs linear pipeline (task A15)",
        "",
        "Same market and the same cached pages and reviews.",
        "",
        "| | Graph | Linear |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {graph[key]} | {linear[key]} |" for name, key in ROWS]
    names = sorted(
        set(graph["complaint_documents_by_company"]) | set(linear["complaint_documents_by_company"])
    )
    lines += [
        "",
        "Complaint documents per competitor:",
        "",
        "| | Graph | Linear |",
        "|---|---|---|",
    ]
    lines += [
        f"| {n} | {graph['complaint_documents_by_company'].get(n, 0)} | "
        f"{linear['complaint_documents_by_company'].get(n, 0)} |"
        for n in names
    ]
    if args.fresh or args.replay:
        lines += ["", "## Cost per run", "", "| Run | Requests | Cache hits | Tokens | Minutes |"]
        lines += ["|---|---|---|---|---|"]
    for label, path in (("fresh", args.fresh), ("replay", args.replay)):
        if path:
            run = out[label] = summarize(path)
            tokens = run["input_tokens"] + run["output_tokens"]
            minutes = (run["runtime_s"] or 0) / 60
            lines.append(
                f"| {label} ({run['engine']}) | {run['requests']} | {run['cache_hits']} | "
                f"{tokens} | {minutes:.1f} |"
            )
    RESULTS.joinpath("graph_vs_linear.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    RESULTS.joinpath("graph_vs_linear.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
