"""Retrieval eval (task C9): Recall@10, MRR@10 and nDCG@10 for BM25, dense, hybrid and
hybrid+rerank, on the labeled queries in eval/retrieval_queries.jsonl over a saved market index.

    uv run python eval/run_retrieval.py              # metrics → eval/results/retrieval.{json,md}
    uv run python eval/run_retrieval.py --pool out.jsonl   # candidates to label, see below

Labels are document-level: a query lists the documents that answer it, so re-chunking never
invalidates them. Each method's chunk ranking is collapsed to documents (first occurrence wins)
before scoring. Relevance is binary.

Labeling (task C8) pools candidates so that no method is favored: the top 10 documents of all
four methods, judged one by one. A document none of the methods returned is unlabeled and counts
as not relevant, as usual for pooled evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from mia import pipeline, runtime
from mia.rag.hybrid import RERANK_DEPTH, HybridIndex
from mia.schemas import Evidence

EVAL_DIR = Path(__file__).parent
QUERIES = EVAL_DIR / "retrieval_queries.jsonl"
RESULTS = EVAL_DIR / "results"
MARKET = "AI meeting assistants"
K = 10
CHUNKS = RERANK_DEPTH  # chunks retrieved per query, then collapsed to documents
METRICS = ("recall@10", "mrr@10", "ndcg@10")

Method = Callable[[str, dict], list[Evidence]]


def methods(index: HybridIndex) -> dict[str, Method]:
    return {
        "bm25": lambda q, f: index.bm25_index.search(q, k=CHUNKS, **f),
        "dense": lambda q, f: index.dense_index.search(q, k=CHUNKS, **f),
        "hybrid": lambda q, f: index.search(q, k=CHUNKS, rerank=False, **f),
        "hybrid+rerank": lambda q, f: index.search(q, k=CHUNKS, rerank=True, **f),
    }


def load_queries(path: Path = QUERIES) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filters(q: dict) -> dict:
    return {"company": q.get("company"), "source_types": q.get("source_types")}


def ranked_docs(hits: list[Evidence]) -> list[str]:
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))


# ---- metrics ------------------------------------------------------------------------------------


def recall(ranked: list[str], relevant: set[str], k: int = K) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str], k: int = K) -> float:
    return next((1 / i for i, d in enumerate(ranked[:k], 1) if d in relevant), 0.0)


def ndcg(ranked: list[str], relevant: set[str], k: int = K) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---- runs ---------------------------------------------------------------------------------------


def evaluate(index: HybridIndex, queries: list[dict]) -> dict:
    per_method: dict[str, dict] = {}
    for name, run in methods(index).items():
        rows, times = [], []
        for q in queries:
            relevant = {r["doc_id"] for r in q["relevant"]}
            start = time.perf_counter()
            ranked = ranked_docs(run(q["query"], filters(q)))
            times.append(time.perf_counter() - start)
            rows.append(
                {
                    "id": q["id"],
                    "recall@10": recall(ranked, relevant),
                    "mrr@10": mrr(ranked, relevant),
                    "ndcg@10": ndcg(ranked, relevant),
                }
            )
        per_method[name] = {
            "mean": {m: round(statistics.mean(r[m] for r in rows), 3) for m in METRICS},
            "median_latency_ms": round(statistics.median(times) * 1000),
            "queries": rows,
        }
    return per_method


def pool(index: HybridIndex, queries: list[dict], depth: int = K) -> list[dict]:
    """For each query, the union of every method's top `depth` documents, best rank first."""
    by_doc: dict[str, list] = {}
    for ch in index.chunks:
        by_doc.setdefault(ch.doc_id, []).append(ch)
    out = []
    for q in queries:
        best: dict[str, int] = {}
        for run in methods(index).values():
            for rank, doc in enumerate(ranked_docs(run(q["query"], filters(q)))[:depth]):
                best[doc] = min(best.get(doc, rank), rank)
        for doc in sorted(best, key=best.get):
            first = by_doc[doc][0]
            text = " ".join(" ".join(c.text.split("\n", 1)[-1].split()) for c in by_doc[doc])
            out.append(
                {
                    "query_id": q["id"],
                    "query": q["query"],
                    "doc_id": doc,
                    "url": first.url,
                    "company": first.company,
                    "source_type": first.source_type,
                    "text": text[:700],
                }
            )
    return out


def as_markdown(results: dict, n_queries: int, index_name: str) -> str:
    lines = [
        f"# Retrieval eval: {index_name}",
        "",
        f"{n_queries} labeled queries, document-level, k={K} (`eval/run_retrieval.py`).",
        "",
        "| Method | Recall@10 | MRR@10 | nDCG@10 | Median latency |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r["mean"]
        lines.append(
            f"| {name} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} "
            f"| {r['median_latency_ms']} ms |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--market", default=MARKET)
    parser.add_argument("--pool", type=Path, help="Write labeling candidates to this JSONL file.")
    args = parser.parse_args()

    runtime.configure(offline=True)  # the eval reads the saved index; it never fetches
    slug, index = pipeline.load_index(args.market)
    queries = load_queries()
    if args.pool:
        rows = pool(index, queries)
        args.pool.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} candidates for {len(queries)} queries → {args.pool}")
        return

    results = evaluate(index, queries)
    RESULTS.mkdir(exist_ok=True)
    summary = {
        "index": slug,
        "chunks": len(index.chunks),
        "queries": len(queries),
        "k": K,
        "results": results,
    }
    (RESULTS / "retrieval.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = as_markdown(results, len(queries), slug)
    (RESULTS / "retrieval.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
