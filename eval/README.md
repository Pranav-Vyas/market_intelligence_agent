# Evaluation

| File | What |
|---|---|
| `gold/competitors_*.json` | Gold competitor lists for discovery recall |
| `retrieval_queries.jsonl` | 36 retrieval queries with document-level relevance labels (task C8) |
| `run_retrieval.py` | Recall@10, MRR@10, nDCG@10 for BM25 / dense / hybrid / hybrid+rerank (task C9) |
| `results/retrieval.{json,md}` | Latest retrieval numbers, per method and per query |
| `tune_retrieval.py`, `results/retrieval_tuning.{json,md}` | Chunk size, RRF candidates and rerank depth, one at a time (task C13) |
| `results/pipeline_runs.json` | Full-run statistics: documents, complaints, LLM requests, runtime |

## Retrieval

```bash
uv run mia snapshot restore <snapshot.zip>      # or build the index with `uv run mia index ...`
uv run python eval/run_retrieval.py             # → results/retrieval.{json,md}
```

The eval reads the saved index of the demo market offline and never fetches anything.

**Queries.** There are 36 queries in three kinds:
- 18 complaint themes, worded differently from the reviews themselves ("the note taker bot joined meetings I never invited it to").
- 12 product facts (SOC 2, Slack, languages, per-seat prices, ...).
- 6 queries with a company or source-type filter, which exercise the filters.

**Labels.** Labels are per document (`doc_id`, with its URL for reading), so re-chunking never breaks them. An App Store review is its own document even though every review of an app shares the app's URL, which is why labels use `doc_id` rather than URL alone. Relevance is binary.

**How the labels were made.**
- **Complaint queries:** candidates were pooled from the top 10 documents of all four methods (`run_retrieval.py --pool candidates.jsonl`), so no method is favored, and each candidate was judged by reading it.
- **Product-fact queries:** the full text of every official and pricing page was also searched for the fact (e.g. "SOC 2", "Slack"), and each match was read. These labels also include relevant pages that no method returned.
- **AI assistance:** the judgments were drafted with AI assistance. **Spot-check them before quoting the numbers** (e.g. 5 random queries), and fix any label you disagree with.

**Reading the numbers.**
- Some product-fact queries have more than 10 relevant pages (q24 "post meeting notes to Slack" has 20), so their Recall@10 can't reach 1. nDCG@10 accounts for this.
- Unlabeled documents count as not relevant.
- Latency is the median per query on a laptop CPU, after the models have loaded.

**When the corpus changes** (re-collection, new competitors), rebuild the candidate pool and re-judge before comparing numbers. Old labels may point at documents that no longer exist.

## Retrieval tuning (task C13)

```bash
uv run python eval/tune_retrieval.py            # → results/retrieval_tuning.{json,md}
```

The script reads the demo market's documents from the cache. It varies one setting at a time
from the defaults (300-word chunks with 45 words of overlap, 50 RRF candidates per retriever,
rerank depth 30). Results from 2026-10-19:

- **Chunk size:** 200-word chunks scored best for every method, but only slightly: nDCG@10
  went up by 0.016 to 0.038 (hybrid+rerank 0.725 → 0.736). 450-word chunks helped the reranker
  (0.738) but hurt plain hybrid.
- **RRF candidates:** 50 is as good as 20 or 100 after reranking. 100 lets more noise into the
  reranker (0.695).
- **Rerank depth:** 30 is best. 10 and 20 miss relevant documents (0.667, 0.712). 50 is slower
  (2.6 s per query) and no better (0.716).

**Decision: keep the defaults for now.** The gains from 200-word chunks are small for 36 queries
tuned and tested on the same set. The same chunks also feed complaint extraction and
verification, so changing them would re-extract every complaint (about 20 requests) and
invalidate the shared snapshot. Revisit when the snapshot is next refreshed (task B16), and
re-label first if the corpus has changed.
