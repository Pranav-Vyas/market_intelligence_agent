"""Retrieval tuning (task C13): how chunk size, the candidates fed to RRF and the rerank depth
change Recall@10, MRR@10 and nDCG@10 on the labeled queries.

    uv run python eval/tune_retrieval.py      # → eval/results/retrieval_tuning.{json,md}

It reads the demo market's documents from the cache (offline: run it after a run or a snapshot
restore) and varies one setting at a time from the current defaults. The labels are per
document, so they stay valid when the chunks change.

Caveat: these are the same 36 queries the defaults are reported on, so a setting that wins
here by a hair may just fit these queries. Prefer clear wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_retrieval as ev  # noqa: E402

from mia import pipeline, runtime  # noqa: E402
from mia.rag import chunking, hybrid  # noqa: E402
from mia.rag.hybrid import HybridIndex  # noqa: E402

CHUNK_SIZES = [(200, 30), (chunking.TARGET_WORDS, chunking.OVERLAP_WORDS), (450, 68)]
CANDIDATES = [20, hybrid.CANDIDATES, 100]
RERANK_DEPTHS = [10, 20, hybrid.RERANK_DEPTH, 50]


def build(docs, target: int, overlap: int) -> HybridIndex:
    index = HybridIndex()
    index.add_chunks([ch for d in docs for ch in chunking.chunk_document(d, target, overlap)])
    return index


def score(index: HybridIndex, queries, methods: list[str], k: int = ev.CHUNKS) -> dict:
    old, ev.CHUNKS = ev.CHUNKS, k
    try:
        results = ev.evaluate(index, queries)
    finally:
        ev.CHUNKS = old
    return {
        m: {**results[m]["mean"], "latency_ms": results[m]["median_latency_ms"]} for m in methods
    }


def main() -> None:
    runtime.configure(offline=True)
    competitors = pipeline.discover_competitors(ev.MARKET, k=runtime.settings().max_competitors)
    docs = pipeline.collect(ev.MARKET, competitors)
    queries = ev.load_queries()
    labeled = {r["doc_id"] for q in queries for r in q["relevant"]}
    missing = labeled - {d.id for d in docs}
    print(f"{len(docs)} documents; {len(missing)} labeled documents missing from the corpus")

    rows = []
    for target, overlap in CHUNK_SIZES:
        index = build(docs, target, overlap)
        result = score(index, queries, ["bm25", "dense", "hybrid", "hybrid+rerank"])
        setting = f"chunks {target} words, {overlap} overlap"
        rows.append({"setting": setting, "chunks": len(index.chunks), **_flat(result)})
        print(rows[-1]["setting"], "done")
    base = build(docs, chunking.TARGET_WORDS, chunking.OVERLAP_WORDS)
    for n in CANDIDATES:
        hybrid.CANDIDATES, old = n, hybrid.CANDIDATES
        try:
            result = score(base, queries, ["hybrid", "hybrid+rerank"])
        finally:
            hybrid.CANDIDATES = old
        rows.append({"setting": f"RRF candidates {n} per retriever", **_flat(result)})
        print(rows[-1]["setting"], "done")
    for depth in RERANK_DEPTHS:
        hybrid.RERANK_DEPTH, old = depth, hybrid.RERANK_DEPTH
        try:
            result = score(base, queries, ["hybrid+rerank"], k=depth)
        finally:
            hybrid.RERANK_DEPTH = old
        rows.append({"setting": f"rerank depth {depth}", **_flat(result)})
        print(rows[-1]["setting"], "done")

    out = ev.RESULTS
    out.mkdir(exist_ok=True)
    defaults = {
        "chunk_words": chunking.TARGET_WORDS,
        "overlap_words": chunking.OVERLAP_WORDS,
        "rrf_candidates": hybrid.CANDIDATES,
        "rerank_depth": hybrid.RERANK_DEPTH,
    }
    summary = {"queries": len(queries), "documents": len(docs), "defaults": defaults, "rows": rows}
    text = json.dumps(summary, indent=2) + "\n"
    (out / "retrieval_tuning.json").write_text(text, encoding="utf-8")
    (out / "retrieval_tuning.md").write_text(as_markdown(summary), encoding="utf-8")
    print(as_markdown(summary))


def _flat(result: dict) -> dict:
    return {f"{m} {k}": v for m, metrics in result.items() for k, v in metrics.items()}


def as_markdown(summary: dict) -> str:
    lines = [
        "# Retrieval tuning",
        "",
        f"{summary['queries']} labeled queries, {summary['documents']} documents. One setting "
        f"changes at a time from the defaults {summary['defaults']}. nDCG@10 (Recall@10).",
        "",
        "| Setting | BM25 | Dense | Hybrid | Hybrid+rerank | Rerank latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        cells = []
        for m in ("bm25", "dense", "hybrid", "hybrid+rerank"):
            nd, rc = r.get(f"{m} ndcg@10"), r.get(f"{m} recall@10")
            cells.append(f"{nd:.3f} ({rc:.3f})" if nd is not None else "—")
        latency = r.get("hybrid+rerank latency_ms")
        lines.append(f"| {r['setting']} | " + " | ".join(cells) + f" | {latency} ms |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
