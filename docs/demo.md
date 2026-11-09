# Demo video script (2–3 minutes)

Record from the snapshot, so nothing depends on live quota:
`uv run mia snapshot restore <zip>`, then `uv run streamlit run app/streamlit_app.py`.

| Time | Show | Say |
|---|---|---|
| 0:00 | README, top | "Give it a market. It finds the competitors, reads their sites and customer complaints, and writes a cited report of market gaps." |
| 0:15 | Terminal: `uv run mia run "AI meeting assistants" --offline` | "Five agents run as a graph. This replays a real run from the cache, so it needs no API calls." |
| 0:35 | App: Competitors, then Features and Pricing | "Every cell links to the vendor's own page. A price is only kept if it's on the page." |
| 1:00 | App: Complaints | "229 complaints, each quoted word for word, grouped into pain points." |
| 1:20 | App: Gaps, open the top card | "Gaps are scored in code, never by the model. Each quote passed a verbatim check and a second model's check. Crashes and billing problems are listed apart: they're failures, not gaps." |
| 1:50 | App: Evidence, search "the bot joined meetings I never invited it to" | "Hybrid search over everything collected: keywords, embeddings, then a reranker." |
| 2:10 | App: Agent trace, competitor Fathom | "Each scout's decisions are logged. Fathom has almost no public complaints, so its scout stopped after two empty searches." |
| 2:30 | eval/results/summary.md | "We report every number, including the aims we missed: discovery recall and quote faithfulness." |
| 2:45 | Market picker: the second market | "The second market ran with no market-specific code." |
