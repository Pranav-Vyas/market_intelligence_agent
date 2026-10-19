# Retrieval tuning

36 labeled queries, 364 documents. One setting changes at a time from the defaults {'chunk_words': 300, 'overlap_words': 45, 'rrf_candidates': 50, 'rerank_depth': 30}. nDCG@10 (Recall@10).

| Setting | BM25 | Dense | Hybrid | Hybrid+rerank | Rerank latency |
|---|---|---|---|---|---|
| chunks 200 words, 30 overlap | 0.660 (0.586) | 0.617 (0.603) | 0.696 (0.648) | 0.736 (0.663) | 1656 ms |
| chunks 300 words, 45 overlap | 0.644 (0.576) | 0.579 (0.554) | 0.672 (0.633) | 0.725 (0.650) | 1614 ms |
| chunks 450 words, 68 overlap | 0.627 (0.564) | 0.580 (0.560) | 0.671 (0.629) | 0.738 (0.674) | 1462 ms |
| RRF candidates 20 per retriever | — | — | 0.682 (0.662) | 0.720 (0.645) | 1409 ms |
| RRF candidates 50 per retriever | — | — | 0.672 (0.633) | 0.725 (0.650) | 1430 ms |
| RRF candidates 100 per retriever | — | — | 0.674 (0.640) | 0.695 (0.623) | 1446 ms |
| rerank depth 10 | — | — | — | 0.667 (0.596) | 443 ms |
| rerank depth 20 | — | — | — | 0.712 (0.637) | 888 ms |
| rerank depth 30 | — | — | — | 0.725 (0.650) | 1402 ms |
| rerank depth 50 | — | — | — | 0.716 (0.630) | 2589 ms |
