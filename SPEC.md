# Market Gap Scout: Build Spec & 10-Day Plan

> SaaS competitor intelligence & market-gap agent (full idea in [Topic.md](Topic.md)).
> Team of 3 · 10 working days (Days 7 and 9 absorb slippage) · Python · repo `Pranav-Vyas/market_intelligence_agent`

---

## 0. TL;DR

- **Input:** a market name, e.g. `"AI meeting assistants"`.
- **Output:** a report with citations covering competitors, a feature matrix, a pricing table, complaint clusters and ranked market gaps, plus a Streamlit app for exploring it.
- **How it works:** An agent decides what to research. Collectors pull official pages and customer complaints. Hybrid RAG (BM25 + embeddings + reranker) retrieves the evidence. An analysis pipeline extracts, clusters and scores complaints into gaps, and every claim is checked against a verbatim quote.
- **Team model:** Each person owns a module and commits their own code. The contracts (schemas and function signatures) are fixed on Day 1, so all three people can work in parallel from that point on.
- **Three rules that keep this on schedule:**
  1. By the end of Day 1, `main` has a **walking skeleton**: the whole pipeline runs end to end on fake data.
  2. **Everything is cached.** Every fetched page and every LLM call is cached. A **frozen data snapshot** exists by the end of Day 4, so scraping problems can't block anyone after that.
  3. **Merge small, merge daily.** **Feature freeze at the end of Day 9.** Day 10 is only for docs and the demo.

---

## 1. Scope

### MVP (must ship)
- `mia run "AI meeting assistants"` runs live, and `--offline` runs from the snapshot. Both write `report.md`, `run.json` and `trace.jsonl`.
- 5–8 competitors **discovered automatically** (not hardcoded).
- A feature matrix of 10–15 features, each cell marked ✓ / ~ / ✗ / ? and backed by a citation.
- A pricing table grouped into Free / Pro / Business / Enterprise buckets.
- Complaint clusters showing share of complaints, severity and number of distinct sources.
- 3–5 **ranked opportunities**, each with a score breakdown and verified customer quotes.
- A Streamlit app for browsing a run.
- Evaluation numbers in the README (§10).

### Stretch (only after `v0.9`)
A news/funding section, more markets, deployment (Streamlit Community Cloud or HF Spaces), Docker, tracing (Langfuse), complaint trends over time, and an open-model comparison (for example Apertus vs Gemini on complaint extraction accuracy, via a second backend in `llm.py`).

### Demo markets
- **Primary:** `AI meeting assistants`. It matches Topic.md and has lots of public complaints.
  - Starter gold list (edit it on Day 1): Otter.ai, Fireflies.ai, Fathom, Granola, tl;dv, Read.ai, Avoma, Fellow, MeetGeek, Notta, Sembly, Krisp, Bluedot.
  - Decide on Day 1 whether platform-bundled tools (Zoom AI Companion, Copilot in Teams, Gemini in Meet) count as competitors, and write the decision down.
- **Secondary (proves the tool generalizes):** `AI calendar / scheduling assistants`. This run happens on Day 8 with **no market-specific code**.

---

## 2. Architecture

```
  "AI meeting assistants"
            │
            ▼
┌───────────────────────────── Agent (orchestrator) ─────────────────────────────┐
│  1 Discover ──► 2 Research loop ──► 3 Analyze ──► 4 Verify loop ──► 5 Report   │
└──────┬──────────────────┬──────────────────────┬──────────────────────┬────────┘
       │ tools            │                      │                      │
       ▼                  ▼                      ▼                      ▼
  Collectors          Hybrid RAG             Analysis                Report
  search, official    chunk → BM25 + dense   features, pricing,      markdown +
  pages, pricing,     → RRF fusion →         complaints → cluster    Streamlit
  Reddit, HN, apps    cross-encoder rerank   → coverage → score
       │                  ▲
       ▼                  │
  data/cache/  (every HTTP fetch + every LLM call; --offline replays it)
```

**Division of labour:** RAG retrieves the evidence. The LLM extracts and reasons over that evidence. The agent decides which research action to take next, and plain code enforces the budgets and the evidence rules.

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language / env | Python 3.12, **uv** (`pyproject.toml` + `uv.lock`) | Everyone gets an identical environment from one command |
| Fetch & clean | `httpx`, `trafilatura`; `playwright` as a fallback only | trafilatura extracts the main text; Playwright handles pricing pages rendered with JavaScript |
| Web search | Tavily API (primary), `ddgs` (no-key fallback) | Tavily is built for LLM pipelines; ddgs needs no key |
| Lexical retrieval | `rank-bm25` | Simple and fast |
| Embeddings | `sentence-transformers`, `BAAI/bge-small-en-v1.5` | Small enough to run well on CPU |
| Vector store | Chroma (local, persistent) | Supports metadata filters (company, source) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast enough on CPU |
| Clustering | scikit-learn `AgglomerativeClustering` with a distance threshold | No need to pick k in advance |
| LLM | `google-genai` (official SDK) behind our own `mia.llm` layer → Gemini free tier | Small dependency footprint; providers are swappable inside `llm.py` |
| Agent | Plain-Python orchestrator + LLM tool calling (a manual loop) | Few dependencies, easy to debug, and every line can be explained in an interview |
| Contracts | Pydantic v2 | One shared source of truth for data shapes |
| CLI / UI | Typer / Streamlit | Quick to build |
| Quality | ruff, pytest, GitHub Actions | CI runs on every PR |

### What runs on your own machine

**No LLM runs locally.** All LLM work goes to the Gemini API. Two small models do run locally, on CPU, with no GPU needed:

| Model | Size | Used for | How it gets there |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | ~130 MB | Embeddings for dense retrieval and complaint clustering | Downloaded automatically from Hugging Face on first use, then cached |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Reranking search results | Same |

Other downloads to expect:
- **PyTorch,** which `sentence-transformers` needs. On Linux the default install pulls the multi-GB CUDA build, so A1 configures uv to use the **CPU-only build** (see uv's PyTorch guide). Then everyone gets the small version from `uv sync`.
- **Playwright's Chromium** (~150 MB, via `uv run playwright install chromium`). Only whoever runs live collection needs it; the offline demo doesn't.

Any laptop with about 8 GB of RAM is enough. Embedding about 2,000 documents takes a few minutes on CPU, and the result is saved, so it's a one-time cost.

**Why not use Gemini's embedding API instead?** It would remove the local models, but it would spend free-tier quota on thousands of chunks and would make offline work and evals depend on the network. The local models are free, unlimited and fast enough.

### LLM usage, keys and quota

**Default: Gemini's free tier (Google AI Studio), so the total cost is $0.** Every LLM call goes through `mia.llm`, our own thin layer over Google's official `google-genai` SDK. Model strings look like `gemini/<model-id>` and live in `.env`. Adding another provider (for example an OpenAI-compatible endpoint serving Apertus) means adding one backend class to `llm.py`; no other code changes.

We don't use LiteLLM. Two of its PyPI releases were hijacked in March 2026 to steal credentials, and with one provider we don't need it.

| Job | Setting (`.env`) | Free default |
|---|---|---|
| Bulk extraction (complaints, pricing, feature cells), LLM judges, cluster names | `MIA_MODEL_BULK` | Gemini **Flash-Lite** (the largest free daily quota) |
| Agent planner, feature taxonomy, final synthesis | `MIA_MODEL_SMART` | Gemini **Flash** |

- **Keys:** each person creates their own free key in Google AI Studio and puts it in `.env` as `GEMINI_API_KEY`. That gives three separate quotas, with no billing setup.
- **The limit is quota, not money.**
  - Free limits apply per minute and per day, and they differ by model.
  - Google has cut them several times. Third-party reports from September 2026 put Flash models anywhere from about 20 to about 1,500 requests per day, with Flash-Lite higher.
  - **On Day 1, check your real limits at aistudio.google.com/rate-limit** and set `MIA_MAX_LLM_CALLS_PER_RUN` from them.
- **Design to stay under quota.** The target is **200 requests or fewer for a fresh full run**:
  - Batch work: 15–25 chunks per extraction call, one call per company for the feature matrix, one call per cluster for coverage, and batched verification.
  - Cache every call (see below), so a rerun uses 0 requests.
  - Cap the planner at `MAX_AGENT_STEPS = 20`.
  - If your Flash daily quota is tiny, point `MIA_MODEL_SMART` at Flash-Lite as well.
- **Privacy:** on the free tier, Google may use prompts and responses to improve its products, and human reviewers may read them. That's acceptable here because everything we send is public web content. Never send keys, private data or anything confidential.
- **Optional upgrade:** if the free planner makes poor decisions or quotas block you, set `MIA_MODEL_SMART` to a paid Gemini model (no code change), or add a backend for another provider. A fresh run would then cost roughly $0.50–2. Only the planner and synthesis need the upgrade; bulk extraction can stay free.
- **How `llm.py` works (built on Day 1):**
  - `llm().extract(prompt, schema=MyModel)` asks Gemini for JSON matching a Pydantic model, validates it, and retries once with the validation error if it doesn't match.
  - `429` and `5xx` errors are retried with exponential backoff, and a per-model throttle keeps requests under `MIA_LLM_RPM`.
  - Requests and tokens are counted per run and saved in `report.stats`.
  - Model IDs change often. `uv run mia models` lists the ones your key can use; put them in `.env`, never in code.
  - Tool calling for the planner gets added on Day 6 (task A13).
- **Every LLM call goes through a disk cache** keyed by `hash(model, system, messages, schema)`. Teammates working on code downstream of an LLM step can replay cached calls offline at no cost. The cache is also part of the frozen snapshot, so the offline demo needs no key at all.

---

## 3. The Agent

The agent is **not a free-form ReAct loop.** It runs in fixed phases, and the LLM makes decisions inside those phases. That gives real agent behaviour while keeping runtime predictable and the demo reliable.

| Phase | What happens | Who decides |
|---|---|---|
| 1. Discover | Searches "best X tools", "X alternatives" and "<seed> vs" → LLM extracts product names and URLs → ranked by how many sources mention them → homepage checked ("is this really a product in this market?") → top K | **Loop:** if fewer than 5 competitors pass the check, the planner writes new queries (at most 2 extra rounds) |
| 2. Research loop | The planner (`MIA_MODEL_SMART`) sees a **coverage table** and the remaining budget, then picks the next tool call | LLM planner; code enforces the budgets |
| 3. Analyze | Index → feature matrix → pricing → complaints → clusters → coverage → scores | Deterministic pipeline |
| 4. Verify loop | Any top-10 opportunity that fails the evidence gates (§4) gets a targeted research round, then is re-scored | LLM planner, at most 2 rounds |
| 5. Report | Markdown report + `run.json` + trace | Deterministic, with LLM-written prose that must cite sources |

**Coverage table** (included in the planner's prompt on every step):
```
company    official  pricing   complaint_docs  complaints   status
Otter      ✓         ✓         41              63           ok
Granola    ✓         ✗ (JS)    6               4            NEEDS: pricing, complaints
```

**Planner tools:** `search_web(query)`, `collect_official(company)`, `collect_complaints(company, source, query?)`, `search_evidence(query, company?, source_types?)`, `fetch_url(url, company, source_type)`, `finish(reason)`.

**Stop rules, enforced in code rather than by the LLM:** the research phase ends when one of these happens:
- every competitor meets its thresholds (`MIN_COMPLAINT_DOCS_PER_CO = 15`, official and pricing collected), or
- the step limit is reached (`MAX_AGENT_STEPS = 20`), or
- the per-run request cap is reached (`MAX_LLM_CALLS_PER_RUN`, set from your free-tier quota).

**Trace:** every step is appended to `trace.jsonl` as `{step, tool, args, result_summary, requests, tokens}`. The UI shows this trace, and it is the most convincing part of the demo.

---

## 4. How a Market Gap Is Found (and Why It's Defensible)

The LLM never invents a gap. Gaps can only come from clusters of complaints that actually exist in the evidence.

```
complaint-type chunks
  → extract (bulk model, batched): {summary, VERBATIM quote, severity 1–3, company}
  → embed summaries → agglomerative clustering (cosine threshold) → LLM names each cluster
  → stats: share, distinct docs, source types, mean severity, companies affected
  → coverage: for each competitor, does its OFFICIAL site address this? (yes / partial / no + quote)
  → score → evidence gates → verify quotes → rank
```

**Severity rubric:**

| Score | Meaning |
|---|---|
| 1 | Annoyance |
| 2 | Blocks a workflow or forces a workaround |
| 3 | Deal-breaker: churn, refund, "switched to X", privacy or security |

**Opportunity score (0–10).** The components are shown in the report and the weights can be tuned in `config.py`.
```
F = cluster.share / max_share                               # relative frequency
S = (mean_severity - 1) / 2                                 # severity, 0..1
G = 1 - coverage          # coverage = (yes + 0.5·partial) / n_competitors
E = min(1, distinct_docs/10) · (0.5 + 0.5·min(1, n_source_types/3))   # evidence strength
score = 10 · (0.35·F + 0.25·S + 0.25·G + 0.15·E)
kind  = "unmet_need" if coverage < 0.4 else "poor_execution"  # vendors claim it, users still complain
```

**Evidence gates.** An opportunity is ranked only if it has:
- at least 5 complaints,
- from at least 3 distinct documents,
- across at least 2 source types.

At most 3 complaints are counted per document, so one viral thread can't dominate. Anything that fails a gate goes under "Weak signals" instead of being ranked.

**Verification** (`verify.py`):
1. **Deterministic check:** each quote must appear verbatim in its chunk (after normalizing whitespace and case). If it doesn't, the complaint is dropped.
2. **LLM judge (bulk model, batched):** "Does this quote support the claim '<cluster label>'?" Each ranked opportunity needs at least 3 quotes that pass.
3. The report publishes the overall **verification rate**.

Target output, in the style of Topic.md:
```
Pain point: Poor Google Calendar integration          kind: poor_execution
Frequency: 18% of complaints (22 complaints, 12 docs, reddit+app_store+hn)
Severity: 2.4 / 3     Coverage: 2/7 competitors address it well
Score: 8.4 / 10  (F 0.9 · S 0.7 · G 0.71 · E 1.0)
Evidence: "[...] it keeps creating duplicate events in GCal [...]" (reddit.com/r/...)  ✓ verified
```

---

## 5. Data Sources

| Source | Used for | How | Priority | If blocked |
|---|---|---|---|---|
| Search API | Discovery; finding pages and threads | Tavily → `ddgs` fallback | P0 | Swap provider |
| Official site | Features and descriptions | httpx + trafilatura; follow nav links containing features / product / integrations / security | P0 | Put the URL manually in `overrides.yaml` |
| Pricing page | Pricing | Same as above, plus Playwright when the extracted text is too short (JS-rendered page) | P0 | Manual URL, otherwise mark pricing "unknown" |
| Reddit | Complaints | Reddit API (PRAW), **if Reddit approves an app** (approval for new apps has been tightened) | P1 | Search results restricted to `site:reddit.com`, title and snippet only. **Reddit's robots.txt disallows fetching its pages and `.json` endpoints, so we don't** (checked Day 1, see `docs/data_sources.md`) |
| Hacker News | Complaints from technical users | Algolia HN Search API (free, no key) | P0 | — |
| App stores | Complaints and ratings | `google-play-scraper`; Apple customer-reviews RSS (JSON), confirmed working on Day 1 | P0 | Skip web-only products |
| Comparison / "X vs Y" articles | Discovery, feature cross-check | Search → trafilatura | P1 | — |
| G2 / Capterra / Trustpilot | Reviews | **Don't scrape.** They use anti-bot measures and their ToS forbids it. Use search-result snippets only. | P2 | Skip |
| News | Context (funding, launches) | Tavily news search | Stretch | — |

**Collection etiquette:**
- Respect `robots.txt`.
- Send at most 1 request per second per domain, with a descriptive User-Agent.
- Use public pages only; never log in to scrape.
- Hash usernames before storing them.
- **Never commit raw scraped data** (see §11.9).

**Caps:**
- At most 40 complaint documents per source per company.
- About 2,000 documents per market in total.
- Each competitor's official site is fetched once and cached.

---

## 6. Repo Layout & Ownership

`A` = Abhay, `B` = data/collection owner, `C` = retrieval/eval/UI owner (roles are explained in §8).

```
market_intelligence_agent/
├── SPEC.md  Topic.md  README.md  AGENTS.md                (A; README shared Day 10)
├── pyproject.toml  uv.lock  .env.example  .gitignore       (A; changes via small PRs only)
├── overrides.yaml                  # manual URL fixes per market (anyone)
├── .github/  CODEOWNERS  pull_request_template.md  workflows/ci.yml   (A)
├── src/mia/
│   ├── schemas.py config.py llm.py runtime.py cli.py       (A)
│   ├── pipeline.py  stubs.py  fake.py (fake.py goes once no stubs are left)  (A)
│   ├── http.py      cache.py                                (B)
│   ├── collectors/  search.py web.py complaints.py reddit.py hn.py appstores.py dedupe.py  (B)
│   ├── rag/         chunking.py bm25.py dense.py hybrid.py rerank.py          (C)
│   ├── discovery.py                                         (A)
│   ├── analysis/
│   │   ├── features.py  pricing.py  complaints.py           (A)
│   │   ├── clustering.py  coverage.py  scoring.py           (B)
│   │   └── verify.py                                        (C)
│   ├── agent/       state.py tools.py planner.py run.py     (A)
│   └── report/      markdown.py                             (A)
├── app/streamlit_app.py                                     (C)
├── eval/  gold/  retrieval_queries.jsonl  run_*.py  results/   (C; B adds extraction evals)
├── tests/ fixtures/  test_*.py                              (each owner tests their module)
├── examples/ai-meeting-assistants.md                        # final showcase report (committed)
└── data/  cache/ snapshots/ indexes/ runs/                  # ALL gitignored
```

---

## 7. Contracts (Agreed on Day 1; Change Them Only by a Dedicated PR)

**The data models live in [`src/mia/schemas.py`](src/mia/schemas.py)**, the single source of truth. Read it before writing any module. Key points:
- `Document` → `Chunk` → `Evidence` flow through retrieval. Every `Complaint` carries a **verbatim** `quote` and its source `url`.
- `Opportunity.evidence` holds **complaint ids**, so the report can show each quote with its source.
- `FeatureCell.status` is yes / partial / no / unknown, and unknown ≠ no.
- `COMPLAINT_SOURCE_TYPES` lists the source types that carry complaints. `doc_id(url, suffix)` makes stable document ids.

```python
# Module boundaries: implement these signatures exactly
# collectors (B)
search(query: str, k: int = 10) -> list[SearchResult]                        # collectors/search.py
fetch_page(url: str, source_type: SourceType, company: str | None) -> Document | None  # web.py
collect_official(c: Competitor) -> list[Document]                            # web.py
collect_complaints(c: Competitor, market: str, source: str,                  # complaints.py
                   query: str | None = None, limit: int = 40) -> list[Document]
#   source is one of COMPLAINT_SOURCES = ("reddit", "hn", "app_store")

# rag (C)
chunk_documents(docs: list[Document]) -> list[Chunk]                         # rag/chunking.py
class HybridIndex:                                                           # rag/hybrid.py
    def add(self, docs: list[Document]) -> None
    def search(self, query: str, k: int = 10, company: str | None = None,
               source_types: list[SourceType] | None = None,
               rerank: bool = True) -> list[Evidence]
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> HybridIndex

# discovery and analysis
discover_competitors(market: str, k: int = 8) -> list[Competitor]                      # A
build_feature_matrix(comps: list[Competitor], index: HybridIndex) -> list[FeatureCell]  # A
extract_pricing(c: Competitor, docs: list[Document]) -> list[PricingTier]              # A
extract_complaints(chunks: list[Chunk]) -> list[Complaint]                             # A
cluster_complaints(complaints: list[Complaint]) -> list[ComplaintCluster]              # B
assess_coverage(cl: ComplaintCluster, comps: list[Competitor],
                index: HybridIndex) -> dict[str, FeatureCell]      # company -> judgment, B
score_opportunities(clusters: list[ComplaintCluster],
                    coverage: dict[str, dict[str, FeatureCell]],   # cluster id -> company -> cell
                    n_competitors: int) -> tuple[list[Opportunity], list[Opportunity]]
                    # (ranked, weak_signals), B
verify(opps: list[Opportunity], complaints: list[Complaint],
       index: HybridIndex) -> list[Opportunity]                                        # C

# pipeline and agent (A)
pipeline.run(market: str, offline: bool = False) -> RunResult      # linear, exists now
agent.run(market: str, offline: bool = False) -> RunResult         # Day 6
```

**Shared plumbing (A), already on `main`:**
- `mia.runtime`: `settings()`, `is_offline()` and `llm()`. Read these instead of passing configuration through every function.
- `mia.llm`: every LLM call goes through `llm().complete(prompt)` or `llm().extract(prompt, schema=MyModel)`. Never call a provider SDK directly; this is what gives everyone caching, the request budget and retries.
- `mia.stubs.stub`: the decorator that marks a fake implementation. Delete it when your real code lands. `mia run` lists the steps that are still stubbed, and the report shows a warning while any are.

On Day 1, A merges **stubs** for every function above. Each stub returns plausible fake data, so the pipeline runs end to end immediately. Each owner then replaces their stubs with real code without breaking `main`.

---

## 8. Roles

| | Owns | Why this split |
|---|---|---|
| **A: Abhay** | Scaffold, schemas, `llm.py`, discovery, feature/pricing/complaint extraction, agent, report, integration | A sets up the shared foundation, so A also owns integrating everyone's modules |
| **B** | HTTP + cache, all collectors, frozen snapshot; then clustering, coverage and scoring | Data first (Days 1–4), then the analytics that run on that data (Days 5–8) |
| **C** | Fixtures, the whole RAG stack, retrieval eval, verification, Streamlit, eval write-up | Retrieval is self-contained and has clear metrics |

Pranav and the third teammate choose B or C at the Day 1 kickoff, based on interest. The repo belongs to Pranav, so whoever is not Pranav needs a collaborator invite (§11.1).

Assumes **about 6–8 focused hours per person per day**, so 10 working days is two working weeks. If you have classes, count *working* days, not calendar days. Days 7 and 9 are partly buffer: if you fall behind, let them absorb it. Never compress Day 1 or Day 10.

---

## 9. Day-by-Day Plan

```
Day 1   Foundation     skeleton on main, contracts agreed, data-access checks     → tag v0.1-skeleton
Day 2   Official data  discovery, official/pricing pages, dense + hybrid search
Day 3   Customer data  Reddit/HN/app stores, reranker, feature matrix, pricing
Day 4   MVP            FROZEN SNAPSHOT, linear pipeline on real data, eval set     → tag v0.2-mvp
Day 5   Analysis       clustering, coverage, scoring, verification, report, UI v1  → tag v0.3-pipeline
Day 6   Agent          planner loop + discovery/verify loops on the working pipeline → tag v0.4-agent
Day 7   Quality        catch-up, tuning on real data, evidence explorer, team demo
Day 8   Prove it       2nd market, all evals, UI v2
Day 9   Harden         bug bash, fixes, FEATURE FREEZE                              → tag v0.9
Day 10  Ship           README, demo video, cleanup, rehearse                        → tag v1.0
```

**How the days are arranged:**
- **Data collection gets three days (2–4).** Scraping is the least predictable part, and everything after it depends on the frozen snapshot.
- **The agent comes on Day 6,** after the whole pipeline already works as a plain linear script. The agent only decides which steps to run, for which company, and when to stop, so building it on working parts keeps it easy to debug.
- **Day 7 exists to catch up and tune,** not to add features.

Every task ID below becomes a GitHub issue. Each PR title includes the task ID.

### Day 1: Foundation
**Kickoff call (1 hour, everyone):**
1. Collaborator invites accepted, repo cloned, `uv sync` works, and everyone has a free Gemini API key in `.env`.
2. Read this spec together and pick roles.
3. Agree on `schemas.py` (A drafts it before the call).
4. Spend 15 minutes writing `eval/gold/competitors_ai_meeting_assistants.json`.
5. A creates the issues.

| ID | Owner | Task | Done when |
|---|---|---|---|
| A1 | A | Scaffold: uv project (CPU-only PyTorch, see §2), ruff, pytest, CI, `.gitignore`, `.env.example`, `AGENTS.md`, CODEOWNERS, PR template, package folders | CI green; **merged by about noon** (everyone else branches from it) |
| A2 | A | `schemas.py` + `config.py` (thresholds, model IDs, paths, score weights) | Merged with or right after A1 |
| A3 | A | `llm.py` on `google-genai`: `complete()`, `extract(schema)`, disk cache, request/token tracker, 429 backoff + throttle, `mia models` | Test with a mocked client passes; one live Gemini call works |
| A4 | A | Walking skeleton: `mia run "<market>"` calls every stub and writes `report.md` with all sections | Report renders on fake data → **tag v0.1-skeleton** |
| B3 | B | **Do this first: access check.** Gemini free-tier limits (from AI Studio, per model), Tavily key, Reddit API, Play/App Store scrapers, HN Algolia. Record the results in `docs/data_sources.md` | Each source marked works / blocked / fallback chosen |
| B1 | B | `http.py` + `cache.py`: polite fetch (timeouts, retries, 1 req/s per domain, robots.txt), `get_or_fetch`, `--offline` | Tests: the second call hits the cache; an offline cache miss raises an error |
| B2 | B | `collectors/search.py`: Tavily + ddgs fallback | `mia search-web "otter.ai alternatives"` prints results |
| C3 | C | **First:** `tests/fixtures/`, about 20 hand-made `Document`s (3 companies, mixed sources) for everyone to develop against | Merged by mid-afternoon |
| C1 | C | `rag/chunking.py`: web pages about 400 tokens with 15% overlap, split on headings; one review or comment = one chunk; contextual header | Tests on fixtures |
| C2 | C | `rag/bm25.py` over chunks | Sensible top-5 on fixtures |

**End-of-day check:** a fresh clone runs `mia run "AI meeting assistants"` and gets a fake report. Everyone has merged at least 1 PR.

### Day 2: Official Data + Hybrid Search
| ID | Owner | Task | Done when |
|---|---|---|---|
| A5 | A | `discovery.py`: listicle / "alternatives" search → LLM extraction → mention ranking → homepage validation → top K | At least 80% of the gold list appears in the top 10 |
| A6 | A | `complaints.py` on fixtures: batched extraction prompt (15–25 chunks per call), severity rubric, verbatim-quote check | Unit tests pass on fixtures; quotes that aren't verbatim are dropped |
| B4 | B | `collectors/web.py`: `collect_official` (homepage, features, integrations, pricing) + Playwright fallback | Official and pricing docs for at least 5 demo competitors |
| C4 | C | `rag/dense.py`: bge-small embeddings + persistent Chroma with metadata | 1k chunks indexed in a few minutes on CPU |
| C5 | C | `rag/hybrid.py`: `HybridIndex` (BM25 top-50 + dense top-50 → RRF with k=60), filters, save/load | The company filter works; the index round-trips through save/load |

**End-of-day check:** official and pricing pages for at least 5 discovered competitors are in the cache.

### Day 3: Customer Data + Extraction
| ID | Owner | Task | Done when |
|---|---|---|---|
| A7 | A | `features.py`: LLM proposes 10–15 canonical features from official pages, then judges each company × feature with a quote (one call per company) | Matrix for at least 5 competitors |
| A8 | A | `pricing.py`: structured extraction. Every price must appear in the page text (regex check) or it is dropped | Pricing table for at least 5 competitors |
| B5 | B | `reddit.py` (or fallback) + `hn.py`: queries like "{name} problem / issue / cancel / alternative" | At least 15 complaint docs for at least 4 competitors |
| B6 | B | `appstores.py` (Google Play + Apple RSS) + `dedupe.py` (URL + normalized-text hash) | Reviews collected for competitors that have apps |
| C6 | C | `rag/rerank.py`: cross-encoder over the hybrid top 30 | Under about 2 seconds per query on CPU |
| C7 | C | Index the real data collected so far; `mia search` CLI with filters; sanity-check results across companies and sources | `mia search "pricing too expensive" --company Otter` returns sensible chunks |

**End-of-day check:** official pages and complaint documents exist for at least 5 competitors.

### Day 4: MVP + Frozen Snapshot
| ID | Owner | Task | Done when |
|---|---|---|---|
| A9 | A | Wire the real modules into a **linear** pipeline (no agent yet) | First real report from the snapshot → **tag v0.2-mvp** |
| A10 | A | Complaint extraction at full scale on the snapshot; count requests against the free quota | At least 200 complaints; request count recorded |
| B7 | B | Full corpus collection for the demo market → **freeze the snapshot** (raw cache + LLM cache) and share it (§11.9) | Teammates can run `--offline` |
| B8 | B | Collection report (documents per company per source, `mia stats`); fill gaps with `overrides.yaml` | No competitor below the minimum thresholds |
| C8 | C | Retrieval eval set: 30–40 queries labeled with **relevant doc URLs** (doc-level, so re-chunking doesn't break the labels) | Labels committed in `eval/` |
| C9 | C | `eval/run_retrieval.py`: Recall@10, MRR@10, nDCG@10 for BM25 / dense / hybrid / hybrid+rerank | Table saved to `eval/results/` |

**End-of-day check:** a rough but real report is generated fully offline from the snapshot.

### Day 5: Analysis (Complete Linear Pipeline)
| ID | Owner | Task | Done when |
|---|---|---|---|
| A11 | A | `report/markdown.py`: sections mirror Topic.md, numbered citations [n] → URL, weak-signals section, short methodology | Report from the snapshot reads well end to end |
| A12 | A | `agent/state.py` + `agent/tools.py`: wrap the existing collector, index and analysis functions as planner tools; compute the coverage table. No LLM loop yet | Unit tests call each tool on the snapshot; coverage table prints correctly |
| B9 | B | `clustering.py`: embed summaries → agglomerative clustering → LLM names clusters → stats | 8–20 coherent clusters; 3 checked by hand |
| B10 | B | `coverage.py` + `scoring.py`: per-competitor coverage judgments from official docs, the score formula with breakdown, evidence gates | Ranked list with breakdowns, plus weak signals |
| C10 | C | `verify.py`: verbatim check + batched LLM support judge + verification rate | Verification rate appears in `report.stats` |
| C11 | C | Streamlit v1: pick a run → tabs for Competitors / Features / Pricing / Complaints / Gaps | Loads the demo run |

**End-of-day check:** `mia run --offline` produces a complete report with ranked, verified opportunities, without the agent → **tag v0.3-pipeline**.

### Day 6: Agent
| ID | Owner | Task | Done when |
|---|---|---|---|
| A13 | A | `agent/planner.py` + `agent/run.py`: planner loop with LLM tool calling, coverage table in the prompt, stop rules, request cap, `trace.jsonl` | Live research for the demo market finishes within 20 steps and the request cap |
| A14 | A | Discovery loop (new queries if fewer than 5 competitors pass validation) + verify loop (weak opportunities → targeted research → re-score) | The trace shows at least one opportunity changing after follow-up research |
| B11 | B | Make collectors agent-safe: query-targeted `collect_complaints`, failures returned as tool errors (never crashes), per-source caps | Agent run survives a blocked site without crashing |
| B12 | B | Tests for clustering and scoring using fixtures where the right answer is known | Tests pass in CI |
| C12 | C | Streamlit: **Agent trace** view (step-by-step tool calls) + score-breakdown charts | Trace of the Day 6 run is browsable |
| C13 | C | Retrieval tuning guided by the eval (chunk size, RRF inputs, rerank depth); re-run the eval | Before/after numbers saved in `eval/results/` |

**End-of-day check:** `mia run "AI meeting assistants"` runs in agent mode live, inside the request cap → **tag v0.4-agent**.

### Day 7: Quality + Catch-Up
- **Morning:** finish anything still open from Days 1–6. That comes before the tasks below.
- **End of day, everyone:** a 30-minute internal demo. Each person runs the full live demo once, and anything that breaks becomes an issue.

| ID | Owner | Task | Done when |
|---|---|---|---|
| A15 | A | Prompt tuning for planner and synthesis; record requests and tokens per fresh and cached run | Fresh run fits comfortably within the free quota |
| B13 | B | Tune clustering and scoring on the real data (distance threshold, gate values, score weights); hand-check the top 5 opportunities | Top 5 look sensible to all three of you |
| C14 | C | Streamlit: **Evidence explorer** (live hybrid search with filters) | Usable for the demo |

### Day 8: Prove It (Second Market + Evals)
| ID | Owner | Task | Done when |
|---|---|---|---|
| A16 | A | Run the second market end to end and fix whatever breaks (no market-specific code) | Second report generated |
| B14 | B | Extraction accuracy: hand-check every pricing tier for 5 companies, plus 30 random feature cells | Accuracy saved in `eval/results/` |
| B15 | B | Discovery recall vs the gold lists (2 markets) + clustering check (20 random assignments) | Numbers saved in `eval/results/` |
| C15 | C | Faithfulness numbers + eval write-up (retrieval table and chart, verification rate) | Ready to paste into the README |
| C16 | C | Streamlit v2: market picker for multiple runs, polish | Both markets browsable |

**End-of-day check:** every row in §10 has a real number.

### Day 9: Harden + Feature Freeze
- **Morning bug bash (everyone):** fresh-clone, run the offline demo and one live run for each market, and file an issue for anything that breaks.
- **Rest of the day:** fix the issues.
- **Stretch:** if the bug list is empty by midday, you may pick *one* stretch item from §1, and only if it can be merged today.

| ID | Owner | Task |
|---|---|---|
| A17 | A | Integration fixes; freeze `config.py` defaults; save the final demo runs for both markets to `examples/` |
| B16 | B | Collector and analysis fixes; refresh the snapshot if anything changed; draft `docs/data_sources.md` |
| C17 | C | UI and RAG fixes; fill test gaps so CI covers each module |

**End of day:** **tag v0.9, FEATURE FREEZE.**

### Day 10: Ship (No New Features)
| ID | Owner | Task |
|---|---|---|
| A18 | A | README: problem, architecture diagram, agent loop, gap method, how to run, design decisions and limitations |
| B17 | B | Cleanup and docstrings in collectors/analysis; finish `docs/data_sources.md` (sources, fallbacks, ethics) |
| C18 | C | Screenshots/GIF, 2–3 minute demo video, eval section of the README |
| ALL | — | Each person gives a 5-minute walkthrough of their module to the other two (interview rehearsal), writes resume bullets with **real numbers from §10**, and the team tags **v1.0** |

---|---|---|
| A17 | A | README: problem, architecture diagram, agent loop, gap method, how to run, design decisions and limitations |
| B15 | B | Cleanup and docstrings in collectors/analysis; finish `docs/data_sources.md` (sources, fallbacks, ethics) |
| C15 | C | Screenshots/GIF, 2–3 minute demo video, eval section of the README |
| ALL | — | Each person gives a 5-minute walkthrough of their module to the other two (interview rehearsal), writes resume bullets with **real numbers from §10**, and the team tags **v1.0** |

---|---|---|
| A14 | A | README: problem, architecture diagram, agent loop, gap method, how to run, design decisions and limitations; commit `examples/ai-meeting-assistants.md` |
| B12 | B | Cleanup and docstrings in collectors/analysis; `docs/data_sources.md` (sources, fallbacks, ethics) |
| C12 | C | Screenshots/GIF, 2–3 minute demo video, eval section of the README |
| ALL | — | Each person gives a 5-minute walkthrough of their module to the other two (interview rehearsal), writes resume bullets with **real numbers from §10**, and the team tags **v1.0** |

---

## 10. Evaluation (This Is What Makes the Project Credible)

| What | How | Aim |
|---|---|---|
| Retrieval | 30–40 labeled queries; Recall@10, MRR@10, nDCG@10 for BM25 / dense / hybrid / hybrid+rerank | hybrid+rerank beats each single method |
| Competitor discovery | Recall@K against the gold list, 2 markets | ≥ 0.8 |
| Pricing extraction | Hand-check every tier of 5 competitors | ≥ 90% exact |
| Feature matrix | Hand-check 30 random cells | ≥ 80% |
| Clustering | 20 random complaint→cluster assignments judged by a person | ≥ 80% sensible |
| Faithfulness | Share of report claims whose quotes pass the verbatim and judge checks | ≥ 95% |
| Cost / latency | LLM requests, tokens and minutes per fresh run and per cached run | Fresh run within the free quota |

These are aims. Report the real numbers even if they miss: an honest table with a short "why" section is worth more than inflated numbers.

---

## 11. Git & Commit Strategy

### 11.1 Getting Access
Git and GitHub are free and don't depend on any AI tool. Each teammate needs a GitHub account, git, and **write access to the repo**.

**Pranav (repo owner), once:**
- Go to Settings → Collaborators and add the other two with **Write** access.
- Go to Settings → Branches → add a rule for `main`: require a pull request, 1 approval and passing status checks, and block force pushes.
- Branch protection is free on public repos. On a private repo it needs GitHub Pro, which is free with the **GitHub Student Developer Pack**. Without protection, the team rule is simply that nobody pushes to `main`.

**Everyone, once per machine:**
```bash
git config --global user.name  "Your Name"
git config --global user.email "an-email-verified-on-your-GitHub-account"   # otherwise commits won't count as yours
# install uv: https://docs.astral.sh/uv/  · install gh: https://cli.github.com/
gh auth login                    # or sign in through VS Code / GitHub Desktop
git clone https://github.com/Pranav-Vyas/market_intelligence_agent.git
cd market_intelligence_agent
uv sync                          # identical dependencies for everyone
cp .env.example .env             # YOUR keys go here; .env is never committed
```
If you'd rather not use the terminal, VS Code's Source Control panel or **GitHub Desktop** can do everything below.

### 11.2 Branching: GitHub Flow
- `main` must **always run** (`uv run pytest` and `mia run --offline` both work).
- Create one short-lived branch per task, named `<firstname>/<area>-<topic>`. Examples: `pranav/collectors-hn`, `abhay/agent-planner`.
- A branch should live for **one day at most**. If a task is bigger than that, split it.

### 11.3 The Daily Loop
```bash
git switch main && git pull                         # start from the latest main
git switch -c pranav/collectors-hn                  # one branch per task
# ...write code, commit as each small piece works...
uv run ruff check . && uv run pytest
git add src/mia/collectors/hn.py tests/test_hn.py   # add specific files, not "git add ." blindly
git commit -m "feat(collectors): add HN Algolia complaint collector"
git push -u origin pranav/collectors-hn
gh pr create --fill --draft                         # open EARLY as a draft so others see progress
# if main moved on while you worked:
git fetch && git merge origin/main                  # resolve conflicts → commit → push
```
Mark the PR "Ready for review" when it's done. Push your branch before you stop for the day, even if the work is unfinished (that's what draft PRs are for).

### 11.4 Commit Messages: Conventional Commits
Format: `type(scope): what changed, in the imperative`.

- **Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `eval`.
- **Scopes:** `collectors`, `rag`, `analysis`, `agent`, `report`, `ui`, `eval`, `llm`, `cli`.

```
feat(rag): add reciprocal rank fusion for BM25 + dense
fix(collectors): fall back to Playwright when pricing text < 500 chars
test(analysis): add clustering fixtures with known groups
eval(rag): label 35 retrieval queries on snapshot v1
```
Commit each time a small step works, typically 3–10 commits a day. Avoid messages like "wip", "fix" or "final final". A good commit history is part of the portfolio.

### 11.5 Pull Requests
- **One task ID per PR.** The title is in conventional-commit style plus the ID, e.g. `feat(rag): hybrid index (C5)`. The description includes `Closes #<issue>`.
- **Keep PRs small:** under about 400 changed lines, not counting lockfiles and fixtures.
- **Merge requirements:** 1 approval from a teammate and green CI (ruff + pytest; tests that need the network are marked `@pytest.mark.network` and skipped in CI).
- **Review within about 3 hours** during working time. Reviewers check that the contract is respected, there are no secrets, the code is readable and there are tests.
- **Merge with "Create a merge commit"** (not squash) and delete the branch afterwards. A merge commit keeps each person's individual commits on `main`, so `git shortlog -sn` and Insights → Contributors show who built what.
- **`.github/pull_request_template.md`:**
  ```markdown
  ## What
  Closes #<issue> (task <ID>)
  ## How I tested
  - [ ] `uv run pytest` passes
  - [ ] Ran: `mia ...` (paste output or a screenshot)
  ## Contract change?
  - [ ] No   - [ ] Yes → schemas.py / signatures changed, teammates pinged in chat
  ```
- **`.github/CODEOWNERS`** automatically requests the right reviewer:
  ```
  *                                   @<A-handle>
  /src/mia/collectors/                @<B-handle>
  /src/mia/http.py                    @<B-handle>
  /src/mia/cache.py                   @<B-handle>
  /src/mia/analysis/clustering.py     @<B-handle>
  /src/mia/analysis/coverage.py       @<B-handle>
  /src/mia/analysis/scoring.py        @<B-handle>
  /src/mia/rag/                       @<C-handle>
  /src/mia/analysis/verify.py         @<C-handle>
  /app/                               @<C-handle>
  /eval/                              @<C-handle>
  ```

### 11.6 Avoiding Merge Conflicts
- **Own directories.** You mostly edit only your own files (§6), which removes most conflicts.
- **Shared files** (`schemas.py`, `config.py`, `pyproject.toml`/`uv.lock`, `cli.py`) are changed only in a tiny dedicated PR. Announce it in chat and merge it fast. Everyone then runs `git merge origin/main` straight away.
- **Adding a dependency:** run `uv add <pkg>`, which updates `pyproject.toml` and `uv.lock` together. If `uv.lock` conflicts, take `main`'s version and run `uv add <your-pkg>` again.
- **Never reformat files you don't own.** Ruff runs in CI with the same configuration for everyone.

### 11.7 Daily Rhythm
| When | What |
|---|---|
| Morning (10 min call) | What I merged yesterday, what I'm doing today, what's blocking me. Swap tasks if someone is stuck. |
| During the day | Open draft PRs early and review each other's PRs within about 3 hours |
| Evening merge window | Get the day's PRs merged. A then runs `mia run --offline` on `main` and posts the result in the group chat. If `main` is broken, fixing it is the first job the next morning. |
| Target | **At least one merged PR per person per day** |

**Stalled tasks:** if a task has no pushed commits by the end of its planned day, the owner says so at the next standup. If it's still not moving a day later, it gets reassigned to whoever has capacity. This keeps one delay from blocking everyone, and it isn't personal.

### 11.8 Milestone Tags
`v0.1-skeleton` (Day 1) → `v0.2-mvp` (Day 4) → `v0.3-pipeline` (Day 5) → `v0.4-agent` (Day 6) → `v0.9` (Day 9, freeze) → `v1.0` (Day 10).
```bash
git switch main && git pull && git tag -a v0.2-mvp -m "Linear pipeline on real data" && git push origin v0.2-mvp
```

### 11.9 What Never Goes in Git
- `.env` and API keys. If a key is ever pushed, **revoke it in the provider's console immediately**. Deleting the commit is not enough.
- Everything under `data/`: the cache, snapshots, indexes and runs. **Share the frozen snapshot as a zip** through Google Drive, or as a GitHub Release asset if the repo is private. Scraped content must not be published in a public repo.
- `.venv/`, `__pycache__/`, notebook outputs (clear them before committing) and model weights.
- What *is* committed: hand-made fixtures, eval gold labels, `eval/results/*.json`, and the final showcase report in `examples/`.

### 11.10 Using AI Coding Tools
- **Free options:**
  - GitHub Copilot Free in VS Code.
  - **Copilot Pro is free for verified students** through GitHub Education. Apply on Day 1, because approval can take a few days.
  - Free tiers of the major chat assistants. Check current limits.
- **Getting good results from a chat assistant.** Paste in:
  1. the task row from §9,
  2. `schemas.py`,
  3. your module's stub and its test,
  4. the error message if there is one.

  Ask for one function at a time, then run it, understand it and commit it.
- **`AGENTS.md`** holds the shared conventions for AI tools: commands, contracts and rules. Many coding assistants read it automatically.
- **Everyone uses their own accounts and keys.** Don't share logins or API keys for any tool.
- **Own your code.** Whatever tool you use, you must be able to explain every line you commit. Interviewers will ask about your module, and your GitHub history should match your resume.

### 11.11 Credit When Pairing
If two people build something together, add a trailer to the commit message after a blank line. GitHub then credits both people.
```
feat(analysis): add opportunity scoring

Co-authored-by: Teammate Name <their-github-email>
```

### 11.12 Fix-It Cheat Sheet
| Problem | Fix |
|---|---|
| Committed on `main` by mistake (not pushed) | `git switch -c you/rescue` (your commits come with you), then `git switch main && git reset --hard origin/main` |
| Merge conflict | `git fetch && git merge origin/main` → resolve in VS Code's merge editor → `git add <files>` → `git commit` → push |
| Push rejected on your branch | `git pull` then `git push` (never `--force` on `main`) |
| Last commit has the wrong author (not pushed) | Fix `git config user.email`, then `git commit --amend --reset-author --no-edit` |
| A secret was pushed | Revoke and rotate the key **first**, then remove it from the code |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Sites block scraping, or pricing pages are rendered with JavaScript | Cache, Playwright fallback, manual URLs in `overrides.yaml`, mark "unknown" instead of guessing |
| No Reddit API access | Search + `.json` fallback; HN and app stores still provide complaints |
| The LLM invents gaps | Gaps come only from clusters; verbatim quote checks; evidence gates; weak signals separated out |
| One viral thread dominates the results | Count distinct documents; at most 3 complaints per document |
| Integration crunch at the end | Walking skeleton on Day 1, daily merges, agent built only on a working pipeline, catch-up on Day 7, freeze on Day 9 |
| A teammate gets blocked | Stubs keep `main` runnable; swap tasks at standup; cut P1 sources first |
| Scope creep | The MVP list in §1; stretch features only after `v0.9` |
| Live demo fails (rate limits, sites change) | Demo from the frozen snapshot and cached LLM calls (`--offline`) |
| Free-tier quota (429 errors, daily caps) | Batched calls, LLM cache, one key per person, `MAX_LLM_CALLS_PER_RUN`, offline snapshot for the demo; if it's still blocking, upgrade only `MIA_MODEL_SMART` to a paid model |

---

## 13. Final Deliverables Checklist
- [ ] `v1.0` tagged; CI green; `uv sync && mia run "AI meeting assistants" --offline` works from a fresh clone (with the snapshot)
- [ ] README: problem, architecture diagram, agent loop, gap method, eval table, how to run, limitations
- [ ] `examples/ai-meeting-assistants.md` (plus the second market's report)
- [ ] Streamlit app with evidence explorer and agent trace; 2–3 minute demo video
- [ ] Snapshot zip shared privately; `docs/data_sources.md`
- [ ] Each person can explain their module end to end, and has resume bullets with real numbers from the eval
