"""Competitor discovery recall (task B15; SPEC.md §10).

    uv run python eval/run_discovery.py      # → eval/results/discovery.{json,md}

Reruns discovery offline for every market with a gold list, so it needs a run of that market in
the cache and spends no requests. A product counts when its name matches a gold name after
`name_key` normalization ("Otter.ai" = "Otter"), or when the gold name is one word of the found
name ("Motion" in "Motion AI calendar").

The report keeps at most `max_competitors` products, so recall against a longer gold list can't
reach 1 at that cut. Recall@|gold| is the fair overall number; the other columns show the cuts.
"""

from __future__ import annotations

import json
from pathlib import Path

from mia import runtime
from mia.discovery import discover_competitors, name_key

EVAL_DIR = Path(__file__).parent
CUTS = (5, 8, 10)
MAX_LISTED = 30


def matches(found: str, gold: str) -> bool:
    key = name_key(gold)
    return name_key(found) == key or key in {name_key(word) for word in found.split()}


def recall_at(found: list[str], gold: list[str], k: int) -> float:
    top = found[:k]
    return sum(any(matches(f, g) for f in top) for g in gold) / len(gold)


def evaluate(gold_path: Path) -> dict:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    market, gold = data["market"], data["competitors"]
    found = [c.name for c in discover_competitors(market, k=MAX_LISTED)]
    in_gold = [f for f in found if any(matches(f, g) for g in gold)]
    return {
        "market": market,
        "gold": gold,
        "found": found,
        "missed": [g for g in gold if not any(matches(f, g) for f in found)],
        "not_in_gold": [f for f in found if f not in in_gold],
        "recall_at_gold_size": round(recall_at(found, gold, len(gold)), 3),
        "recall_all_found": round(recall_at(found, gold, len(found)), 3),
        **{f"recall_at_{k}": round(recall_at(found, gold, k), 3) for k in CUTS},
        "precision_of_first_8": round(
            sum(f in in_gold for f in found[:8]) / max(len(found[:8]), 1), 3
        ),
    }


def main() -> None:
    runtime.configure(offline=True)
    rows = [evaluate(p) for p in sorted((EVAL_DIR / "gold").glob("competitors_*.json"))]
    lines = [
        "# Competitor discovery (task B15)",
        "",
        "Aim: recall of at least 0.8. Gold lists are hand-written starting lists: see their notes.",
        "",
        "| Market | Gold | Found | Recall@5 | Recall@8 | Recall@10 | Recall@gold size "
        "| Precision, first 8 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['market']} | {len(r['gold'])} | {len(r['found'])} | {r['recall_at_5']} | "
            f"{r['recall_at_8']} | {r['recall_at_10']} | {r['recall_at_gold_size']} | "
            f"{r['precision_of_first_8']} |"
        )
    lines.append("")
    for r in rows:
        lines += [
            f"**{r['market']}.** Found: {', '.join(r['found'])}.",
            f"Missed: {', '.join(r['missed']) or 'none'}. "
            f"Not on the gold list: {', '.join(r['not_in_gold']) or 'none'}.",
            "",
        ]
    out = EVAL_DIR / "results"
    (out / "discovery.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "discovery.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("requests:", runtime.llm().usage.requests, "hits:", runtime.llm().usage.cache_hits)


if __name__ == "__main__":
    main()
