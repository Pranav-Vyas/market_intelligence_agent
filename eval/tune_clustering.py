"""Clustering and scoring tuning (task B13) on the demo market's real complaints.

    uv run python eval/tune_clustering.py [run.json]   # → eval/results/clustering_tuning.{json,md}

Compares linkage methods and distance thresholds, then the LLM refinement step, against
gold/complaint_themes_*.json with B-cubed precision, recall and F1. The themes were drafted with
AI assistance from the summaries alone; spot-check them before quoting the numbers.

It then re-ranks the run's pain points under other score weights and evidence gates, using the
breakdown stored in run.json, to show how much the ranking depends on those choices.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mia import runtime
from mia.analysis.clustering import _cap_per_doc, agglomerative, refine
from mia.rag.dense import embed_texts
from mia.schemas import Complaint

HERE = Path(__file__).parent
GOLD = json.loads((HERE / "gold" / "complaint_themes_ai_meeting_assistants.json").read_text())
THEMES: dict[str, str] = GOLD["themes"]
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MIN_RANKABLE = 5


def bcubed(groups: list[list[str]], universe: list[str]) -> dict[str, float]:
    """B-cubed over complaint summaries. Complaints in no group count as their own group."""
    groups = groups + [[s] for s in _leftover(groups, universe)]
    sizes = Counter(THEMES[s] for s in universe)
    precision, recall = [], []
    for g in groups:
        inside = Counter(THEMES[s] for s in g)
        for s in g:
            same = inside[THEMES[s]]
            precision.append(same / len(g))
            recall.append(same / sizes[THEMES[s]])
    p, r = float(np.mean(precision)), float(np.mean(recall))
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3)}


def _leftover(groups: list[list[str]], universe: list[str]) -> list[str]:
    left = Counter(universe)
    left.subtract(Counter(s for g in groups for s in g))
    return [s for s, n in left.items() for _ in range(n)]


def describe(groups: list[list[str]], universe: list[str]) -> dict:
    big = [g for g in groups if len(g) >= MIN_RANKABLE]
    pure = [Counter(THEMES[s] for s in g).most_common(1)[0][1] / len(g) for g in big]
    return {
        **bcubed(groups, universe),
        "clusters_of_5_or_more": len(big),
        "complaints_in_them": sum(len(g) for g in big),
        "purity_of_them": round(float(np.mean(pure)), 3) if pure else None,
        "largest": max((len(g) for g in groups), default=0),
    }


def complete_linkage(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, np.inf)
    members = [[i] for i in range(len(vectors))]
    while True:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if not dist[i, j] <= threshold:
            break
        i, j = min(i, j), max(i, j)
        merged = np.maximum(dist[i], dist[j])
        dist[i], dist[:, i] = merged, merged
        dist[i, i] = np.inf
        dist[j], dist[:, j] = np.inf, np.inf
        members[i] += members[j]
        members[j] = []
    return [m for m in members if m]


WEIGHTS = {
    "default (F .35 S .25 G .25 E .15)": {"F": 0.35, "S": 0.25, "G": 0.25, "E": 0.15},
    "more gap (F .25 S .25 G .35 E .15)": {"F": 0.25, "S": 0.25, "G": 0.35, "E": 0.15},
    "more frequency (F .45 S .2 G .2 E .15)": {"F": 0.45, "S": 0.2, "G": 0.2, "E": 0.15},
    "more severity (F .25 S .35 G .25 E .15)": {"F": 0.25, "S": 0.35, "G": 0.25, "E": 0.15},
    "equal (.25 each)": {"F": 0.25, "S": 0.25, "G": 0.25, "E": 0.25},
}
GATES = {
    "default (5 complaints, 3 docs, 2 companies, 2 sources)": (5, 3, 2, 2, 0),
    "no company gate": (5, 3, 1, 2, 0),
    "no source gate": (5, 3, 2, 1, 0),
    "2 sources or 3 companies": (5, 3, 2, 2, 3),
    "stricter (8 complaints, 5 docs)": (8, 5, 2, 2, 0),
}


def sensitivity(report: dict) -> dict:
    """Market gaps under other weights and gates. Weak signals have a breakdown too, so a gate
    change can rank them; their quotes wouldn't have been verified yet."""
    clusters = {c["id"]: c for c in report["clusters"]}
    needs = [
        o
        for o in report["opportunities"] + report["weak_signals"]
        if o["kind"] != "product_failure"
    ]

    def passes(o: dict, gate: tuple[int, int, int, int, int]) -> bool:
        cl = clusters[o["cluster_id"]]
        complaints, docs, companies, sources, or_companies = gate
        corroborated = len(cl["source_types"]) >= sources or (
            or_companies and len(cl["companies"]) >= or_companies
        )
        return (
            len(cl["complaint_ids"]) >= complaints
            and cl["distinct_docs"] >= docs
            and len(cl["companies"]) >= companies
            and corroborated
        )

    default_gate = GATES["default (5 complaints, 3 docs, 2 companies, 2 sources)"]
    by_weights = {}
    for name, w in WEIGHTS.items():
        kept = [o for o in needs if passes(o, default_gate)]
        kept.sort(key=lambda o: -sum(w[k] * v for k, v in o["score_breakdown"].items()))
        by_weights[name] = [o["title"] for o in kept]
    by_gates = {}
    default = WEIGHTS["default (F .35 S .25 G .25 E .15)"]
    for name, gate in GATES.items():
        kept = [o for o in needs if passes(o, gate)]
        kept.sort(key=lambda o: -sum(default[k] * v for k, v in o["score_breakdown"].items()))
        by_gates[name] = [o["title"] for o in kept]
    return {"weights": by_weights, "gates": by_gates}


def main() -> None:
    runtime.configure()
    s = runtime.settings()
    default = sorted(s.runs_dir.glob("ai-meeting-assistants/*/run.json"))[-1]
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    report = json.loads(run.read_text(encoding="utf-8"))
    complaints = [Complaint.model_validate(c) for c in report["complaints"]]
    complaints = [c for c in complaints if c.summary in THEMES]
    universe = [c.summary for c in complaints]
    vectors = np.array(embed_texts(universe), dtype=float)

    rows = []
    for name, method in (("average", agglomerative), ("complete", complete_linkage)):
        for t in THRESHOLDS:
            groups = [[universe[i] for i in g] for g in method(vectors, t)]
            rows.append({"method": name, "threshold": t, **describe(groups, universe)})

    seeds = [[complaints[i] for i in g] for g in agglomerative(vectors, s.cluster_distance)]
    refined = refine(seeds)
    groups = [[c.summary for c in members] for _, _, members in refined]
    rows.append(
        {
            "method": "average + LLM refinement",
            "threshold": s.cluster_distance,
            **describe(groups, universe),
        }
    )
    failures = set(GOLD["failures"])
    agree = total = 0
    for _, category, members in refined:
        for c in members:
            total += 1
            agree += (category == "failure") == (THEMES[c.summary] in failures)

    detail = [
        {
            "label": label,
            "category": category,
            "size": len(_cap_per_doc(members, s.max_complaints_per_doc)),
            "themes": Counter(THEMES[c.summary] for c in members).most_common(3),
            "companies": sorted({c.company for c in members if c.company}),
            "sources": sorted({c.source_type for c in members}),
        }
        for label, category, members in sorted(refined, key=lambda g: -len(g[2]))
    ]
    results = {
        "run": str(run.parent.name),
        "complaints": len(universe),
        "rows": rows,
        "category_agreement": round(agree / total, 3) if total else None,
        "refined_clusters": detail,
        "sensitivity": sensitivity(report),
    }
    out = HERE / "results"
    (out / "clustering_tuning.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        f"# Clustering tuning (task B13): {len(universe)} complaints, run {results['run']}",
        "",
        "| Method | Threshold | B³ precision | B³ recall | B³ F1 | Clusters ≥5 | Complaints "
        "in them | Their purity | Largest |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['threshold']} | {r['precision']} | {r['recall']} | {r['f1']} | "
            f"{r['clusters_of_5_or_more']} | {r['complaints_in_them']} | {r['purity_of_them']} | "
            f"{r['largest']} |"
        )
    lines += [
        "",
        f"Failure/need category agrees with the drafted themes on {results['category_agreement']}"
        " of the complaints the refinement kept.",
        "",
    ]
    for title, rows_of in (
        ("Market gaps under other score weights", results["sensitivity"]["weights"]),
        ("Market gaps under other evidence gates", results["sensitivity"]["gates"]),
    ):
        lines += [f"## {title}", "", "| Setting | Ranked, best first |", "|---|---|"]
        lines += [f"| {name} | {'; '.join(titles) or 'none'} |" for name, titles in rows_of.items()]
        lines.append("")
    (out / "clustering_tuning.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    for d in detail:
        print(d["size"], d["category"], d["label"], d["themes"], d["companies"], d["sources"])


if __name__ == "__main__":
    main()
