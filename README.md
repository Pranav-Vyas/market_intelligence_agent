# Market Gap Scout

Give it a market such as "AI meeting assistants". It discovers the competitors, collects their
official pages and real customer complaints, and produces a report with citations: a feature
comparison, a pricing comparison, complaint clusters, and ranked market gaps backed by verified
quotes.

> Work in progress. The full plan is in [SPEC.md](SPEC.md). Every pipeline step runs on real data
> now, and `mia run` runs it as a multi-agent graph (`--linear` for the plain pipeline).

## Quick start
```bash
# 1. Install uv: https://docs.astral.sh/uv/
uv sync                          # Python 3.12 + dependencies (no PyTorch needed)
cp .env.example .env             # add your free Gemini key, then:
uv run mia models                # pick model IDs for MIA_MODEL_BULK / MIA_MODEL_SMART in .env

uv run mia run "AI meeting assistants"
```
The report is written to `data/runs/<market>/<timestamp>/report.md`, with the full data in
`run.json`.

Each run also saves its search index, so you can look through the evidence yourself:
```bash
uv run mia search "pricing too expensive" --company Otter      # -s hn / app_store / pricing ...
uv run mia index "AI meeting assistants"   # collect and index without the analysis steps
```

### Offline, from a teammate's snapshot
Collection and LLM calls are cached. A snapshot zips that cache so others can run everything
without network access or quota (share it privately; scraped data never goes in git):
```bash
uv run mia snapshot create                       # → data/snapshots/<name>-<time>.zip
uv run mia snapshot restore path/to/snapshot.zip # prints any .env model settings to match
uv run mia run "AI meeting assistants" --offline
uv run mia stats                                 # documents per competitor and source
```

### Browse a run in the browser
```bash
uv sync --extra ui                         # Streamlit and Plotly (not needed for the CLI)
uv run streamlit run app/streamlit_app.py  # competitors, features, pricing, gaps, agent trace
```

## Development
```bash
uv run pytest                    # offline tests
uv run ruff check . && uv run ruff format .
```
Team conventions are in [AGENTS.md](AGENTS.md) and SPEC.md §11.
