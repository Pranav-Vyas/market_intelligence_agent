# Retrieval eval: ai-meeting-assistants

36 labeled queries, document-level, k=10 (`eval/run_retrieval.py`).

| Method | Recall@10 | MRR@10 | nDCG@10 | Median latency |
|---|---|---|---|---|
| bm25 | 0.576 | 0.800 | 0.644 | 0 ms |
| dense | 0.554 | 0.707 | 0.579 | 16 ms |
| hybrid | 0.633 | 0.792 | 0.672 | 12 ms |
| hybrid+rerank | 0.650 | 0.898 | 0.725 | 1419 ms |
