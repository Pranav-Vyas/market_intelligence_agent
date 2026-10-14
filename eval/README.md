# Evaluation

| File | What |
|---|---|
| `gold/competitors_*.json` | Gold competitor lists for discovery recall |
| `retrieval_queries.jsonl` | 36 retrieval queries with document-level relevance labels (task C8) |
| `run_retrieval.py` | Recall@10, MRR@10, nDCG@10 for BM25 / dense / hybrid / hybrid+rerank (task C9) |
| `results/retrieval.{json,md}` | Latest retrieval numbers, per method and per query |
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
