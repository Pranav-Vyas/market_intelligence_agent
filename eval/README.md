# Evaluation

| File | What |
|---|---|
| `gold/competitors_*.json` | Gold competitor lists for discovery recall |
| `retrieval_queries.jsonl` | 36 retrieval queries with document-level relevance labels (task C8) |
| `run_retrieval.py` | Recall@10, MRR@10, nDCG@10 for BM25 / dense / hybrid / hybrid+rerank (task C9) |
| `results/retrieval.{json,md}` | Latest retrieval numbers, per method and per query |
| `tune_retrieval.py`, `results/retrieval_tuning.{json,md}` | Chunk size, RRF candidates and rerank depth, one at a time (task C13) |
| `results/pipeline_runs.json` | Full-run statistics: documents, complaints, LLM requests, runtime |
| `gold/complaint_themes_*.json` | Each complaint summary's theme, and which themes are failures (task B13) |
| `tune_clustering.py`, `results/clustering_tuning.{json,md}` | Linkage, thresholds and regrouping against those themes; ranking under other weights and gates (task B13) |
| `results/top5_check.md` | Hand check of the top pain points (task B13) |
| `compare_runs.py`, `results/graph_vs_linear.{json,md}` | Graph vs linear, and cost per fresh and cached run (task A15) |

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

## Clustering and scoring tuning (task B13)

```bash
uv run python eval/tune_clustering.py <run.json>    # → results/clustering_tuning.{json,md}
```

**The themes are AI-drafted** from the summaries alone, so spot-check them before quoting F1.
Complaints whose summary has no theme are left out, so a run on a new corpus covers few of them.

- **Linkage:** average beats complete at every useful threshold. Both top out near 0.45 B-cubed F1.
- **Regrouping:** one LLM call on the seeds gives 0.49 on the snapshot run's complaints (0.44
  without it) and 0.52–0.59 on the Day 6 complaints across prompt variations (0.46 without it).
  Precision gains most. Medium thinking was no better than low.
- **Failure or need:** agrees with the themes on about 90% of complaints in the snapshot run, and
  79–94% across prompt variations.
- **Weights:** the same three gaps rank under every variant tried; only their order changes.
- **Gates:** these decide more than the weights do. Dropping the two-source gate ranks forced login
  and forced calendar access, which is a decision for the team (`results/top5_check.md`).

## Graph vs linear and costs (task A15)

```bash
uv run python eval/compare_runs.py <graph run.json> <linear run.json> --fresh <run.json> --replay <run.json>
```

On the same cached pages and reviews, the graph costs about twice the LLM calls of the linear
pipeline and finds the same complaints. Its gain is 5 more official pages, which changed one
coverage judgment. Follow-up research found no new documents on this corpus, though in the fresh
run it lifted one weak signal to a ranked gap. Treat that as one run, not a measured advantage.

A fresh run makes about 103 LLM calls (92 requests, the rest repeats answered by the cache) in
11 minutes, against about 164 calls before tuning, on a newly collected corpus each time. It stays
well inside the cap of 200. Scout decisions now stop when searches run dry, research skips
competitors with nothing left, and product failures skip the coverage call.

gemini-3.8-flash answered none of these requests (503 high demand, then its daily quota), so every
smart call fell back to flash-lite. The results therefore show what flash-lite does.
