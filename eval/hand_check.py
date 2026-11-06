"""Hand checks of extraction and clustering (tasks B14, B15; SPEC.md §10).

    uv run python eval/hand_check.py sheet <run.json> > sheet.md   # what to read and judge
    uv run python eval/hand_check.py score                         # → results/hand_check.md

`sheet` picks the items with a fixed seed, so nobody chooses the easy ones: every pricing tier of
5 competitors, 30 random feature cells and 20 random complaint-to-cluster assignments. Each item
is shown with its source text. Verdicts go in gold/hand_check_<market>.json as
{"pricing": {id: "ok" or "ok: note" or "wrong: why"}, ...}; `score` counts them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from mia import pipeline, runtime
from mia.collectors.complaints import COMPLAINT_SOURCES

EVAL_DIR = Path(__file__).parent
GOLD = EVAL_DIR / "gold"
RESULTS = EVAL_DIR / "results"
SEED = 8
PRICING_COMPANIES = 5
FEATURE_CELLS = 30
ASSIGNMENTS = 20
SHOWN_CHARS = 1800
AIMS = {"pricing": 0.9, "features": 0.8, "clustering": 0.8}


def pick(report: dict) -> dict:
    """The items to check, chosen only by the seed and the order in the report."""
    rng = random.Random(SEED)
    with_prices = list(dict.fromkeys(t["company"] for t in report["pricing"]))
    companies = sorted(rng.sample(with_prices, min(PRICING_COMPANIES, len(with_prices))))
    pricing = [t for t in report["pricing"] if t["company"] in companies]
    cells = rng.sample(report["features"], min(FEATURE_CELLS, len(report["features"])))
    by_id = {c["id"]: c for c in report["complaints"]}
    pairs = [(cl["label"], cid) for cl in report["clusters"] for cid in cl["complaint_ids"]]
    chosen = rng.sample(pairs, min(ASSIGNMENTS, len(pairs)))
    return {
        "companies": companies,
        "pricing": pricing,
        "features": cells,
        "clustering": [{"cluster": label, **by_id[cid]} for label, cid in chosen],
    }


def tier_id(t: dict) -> str:
    return f"{t['company']}|{t['tier']}|{t['bucket']}"


def cell_id(c: dict) -> str:
    return f"{c['company']}|{c['feature']}"


def sheet(run: Path) -> None:
    report = json.loads(run.read_text(encoding="utf-8"))
    items = pick(report)
    _, index = pipeline.load_index(report["market"])
    pages: dict[str, list[str]] = {}
    for ch in index.chunks:
        if ch.source_type not in COMPLAINT_SOURCES:
            pages.setdefault(ch.url, []).append(ch.text)

    print(f"# {report['market']}, run {run.parent.name}\n\n## Pricing: {items['companies']}\n")
    for url in dict.fromkeys(t["source_url"] for t in items["pricing"]):
        print(f"### Page {url}\n" + "\n".join(pages.get(url, ["(page not in the index)"])) + "\n")
        for t in (t for t in items["pricing"] if t["source_url"] == url):
            print(
                f"- `{tier_id(t)}` monthly={t['monthly_usd']} annual={t['annual_monthly_usd']} "
                f"per_seat={t['per_seat']} limits={t['limits']}"
            )
        print()
    print("\n## Feature cells\n")
    for c in items["features"]:
        print(f"### `{cell_id(c)}`: {c['status']}\nQuote: {c['quote']!r}\nURL: {c['url']}")
        print(_company_pages(index, c, pages) + "\n")
    print("\n## Complaint assignments\n")
    for i, c in enumerate(items["clustering"]):
        print(
            f"- [{i}] cluster: {c['cluster']!r} | {c['company']}: {c['summary']!r} | {c['quote']!r}"
        )


def _company_pages(index, cell: dict, pages: dict[str, list[str]]) -> str:
    """For a claim, the page it cites. For "unknown", where the feature's words appear at all."""
    if cell["url"]:
        return " ".join(pages.get(cell["url"], ["(page not in the index)"]))[:SHOWN_CHARS]
    hits = index.search(
        cell["feature"],
        k=3,
        company=cell["company"],
        source_types=["official", "docs", "pricing"],
        rerank=False,
    )
    return "\n".join(f"[{h.chunk.url}] {h.chunk.text[:600]}" for h in hits) or "(no pages at all)"


def score() -> None:
    rows, details = [], []
    for path in sorted(GOLD.glob("hand_check_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        details.append(f"\n## {data['market']} (run {data['run']})\n")
        for kind in ("pricing", "features", "clustering"):
            verdicts = data.get(kind, {})
            if not verdicts:
                continue
            ok = sum(v.split(":")[0] == "ok" for v in verdicts.values())
            rate = ok / len(verdicts)
            rows.append((data["market"], kind, ok, len(verdicts), rate))
            wrong = [(k, v) for k, v in verdicts.items() if v.split(":")[0] != "ok"]
            if wrong:
                details.append(f"**{kind}, not right:**\n")
                details += [f"- {k}: {v.split(':', 1)[-1].strip()}" for k, v in wrong]
                details.append("")
    lines = [
        "# Hand checks (tasks B14, B15)",
        "",
        "Items are picked by `eval/hand_check.py` with a fixed seed. **The verdicts are "
        "AI-drafted, not yet",
        "confirmed by a person:** read the sheet and overrule anything you disagree with.",
        "",
        "| Market | Check | Right | Checked | Rate | Aim |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {market} | {kind} | {ok} | {n} | {rate:.0%} | {AIMS[kind]:.0%} |"
        for market, kind, ok, n, rate in rows
    ]
    text = "\n".join(lines + details) + "\n"
    (RESULTS / "hand_check.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sheet", "score"])
    parser.add_argument("run", type=Path, nargs="?")
    args = parser.parse_args()
    runtime.configure()
    if args.command == "sheet":
        sheet(args.run)
    else:
        score()


if __name__ == "__main__":
    main()
