# AGENTS.md: conventions for humans and AI coding tools

Market Gap Scout: given a market ("AI meeting assistants"), discover competitors, collect official
pages and customer complaints, retrieve evidence with hybrid RAG, and produce a cited report of
features, pricing, complaint clusters and ranked market gaps. Full plan: `SPEC.md`.

## Commands
```bash
uv sync                                   # install (CPU-only PyTorch is preconfigured)
uv run mia run "AI meeting assistants"    # full pipeline → data/runs/<market>/<time>/report.md
uv run mia run "..." --offline            # cached pages and LLM responses only
uv run mia models                         # Gemini model IDs your key can use
uv run pytest                             # tests (network tests skipped; run them with -m network)
uv run ruff check . && uv run ruff format .
```

## Rules
- **Contracts:** data models are in `src/mia/schemas.py`, and function signatures are in SPEC.md §7.
  Don't change either without a small dedicated PR and a heads-up to the team.
- **LLM calls:** always `from mia.runtime import llm`, then `llm().complete(...)` or
  `llm().extract(prompt, schema=Model)`. Never import a provider SDK anywhere else.
- **Settings:** read them with `mia.runtime.settings()`. Never hardcode model IDs, keys or paths.
- **Stubs:** functions decorated with `@stub` return fake data. Remove the decorator when the real
  implementation lands, and keep the function signature unchanged.
- **Tests:** every module gets tests that run offline. Anything that needs the internet or an API
  key is marked `@pytest.mark.network`.
- **Never commit:** `.env`, API keys, or anything under `data/`.
- **Dependencies:** add them with `uv add <pkg>` (it updates `pyproject.toml` and `uv.lock`
  together). Don't add a second LLM SDK; add a backend inside `llm.py` instead.
- **Scraping etiquette:** respect robots.txt, at most 1 request per second per domain, public pages
  only, hash usernames before storing them.

## Git
- Branch per task: `<firstname>/<area>-<topic>`, e.g. `pranav/collectors-hn`.
- Conventional commits: `feat(rag): add RRF fusion`. Scopes: collectors, rag, analysis, agent,
  report, ui, eval, llm, cli.
- PR per task ID, at most about 400 changed lines, one approval, green CI, merged with a merge
  commit.
