# Market Gap Scout

Give it a market such as "AI meeting assistants". It discovers the competitors, collects their
official pages and real customer complaints, and produces a report with citations: a feature
comparison, a pricing comparison, complaint clusters, and ranked market gaps backed by verified
quotes.

> Work in progress. The full plan is in [SPEC.md](SPEC.md). Steps that still return fake data are
> listed when you run the pipeline.

## Quick start
```bash
# 1. Install uv: https://docs.astral.sh/uv/
uv sync                          # Python 3.12 + dependencies (CPU-only PyTorch)
cp .env.example .env             # add your free Gemini key, then:
uv run mia models                # pick model IDs for MIA_MODEL_BULK / MIA_MODEL_SMART in .env

uv run mia run "AI meeting assistants"
```
The report is written to `data/runs/<market>/<timestamp>/report.md`, with the full data in
`run.json`.

## Development
```bash
uv run pytest                    # offline tests
uv run ruff check . && uv run ruff format .
```
Team conventions are in [AGENTS.md](AGENTS.md) and SPEC.md §11.
