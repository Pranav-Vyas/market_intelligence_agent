# Retrieval eval: ai-meeting-assistants

36 labeled queries, document-level, k=10 (`eval/run_retrieval.py`).

| Method | Recall@10 | MRR@10 | nDCG@10 | Median latency |
|---|---|---|---|---|
| bm25 | 0.560 | 0.771 | 0.620 | 1 ms |
| dense | 0.547 | 0.736 | 0.580 | 78 ms |
| hybrid | 0.625 | 0.850 | 0.675 | 74 ms |
| hybrid+rerank | 0.639 | 0.856 | 0.701 | 1389 ms |
