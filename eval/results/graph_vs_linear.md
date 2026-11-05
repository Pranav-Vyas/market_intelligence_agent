# Multi-agent graph vs linear pipeline (task A15)

Same market and the same cached pages and reviews.

| | Graph | Linear |
|---|---|---|
| Competitors | 8 | 8 |
| Complaints | 229 | 229 |
| Pain points | 19 | 19 |
| Ranked market gaps | 3 | 3 |
| Verified market gaps | 3 | 3 |
| Product failures | 6 | 6 |
| Weak signals | 10 | 10 |
| Quote pass rate | 0.902 | 0.902 |
| LLM calls from an empty cache | 92 | 48 |

Complaint documents per competitor:

| | Graph | Linear |
|---|---|---|
| Avoma | 7 | 7 |
| Fathom | 3 | 3 |
| Fireflies | 62 | 62 |
| Granola | 80 | 80 |
| Krisp | 53 | 53 |
| Otter | 79 | 79 |
| Read.ai | 26 | 26 |
| tl;dv | 12 | 12 |

## Cost per run

| Run | Requests | Cache hits | Tokens | Minutes |
|---|---|---|---|---|
| replay (graph) | 0 | 92 | 0 | 0.3 |

## Fresh runs

Not in the table above, because the corpus differs. Before the scouts moved to the bulk model: 92 requests and 11 cache hits, 332,121 tokens, 11.4 minutes. During a Flash-Lite overload, the same evening: 62 minutes and 40 failed calls, so it is not a cost measurement. See `pipeline_runs.json`.
