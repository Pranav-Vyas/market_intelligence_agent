# Data sources

What each collector uses, whether it works, and the fallback. Re-check any time with:

```bash
uv run mia check-access            # or --markdown to paste a fresh table below
```

## Access check (2026-10-10, from a home connection, no keys set)

| Source | Status | Detail |
|---|---|---|
| Gemini API | no key | Set GEMINI_API_KEY in .env (needed from Day 2) |
| Tavily search | no key | Optional: set TAVILY_API_KEY; ddgs is used without it |
| ddgs search (no key) | works | 3 results |
| Hacker News (Algolia API) | works | 132 matching comments |
| Reddit official API | no key | Register a script app at reddit.com/prefs/apps; new apps may need Reddit's approval |
| Reddit public JSON | blocked | robots.txt disallows `https://www.reddit.com/search.json` |
| Apple App Store reviews (RSS) | works | 50 recent reviews for Otter Transcribe Voice Notes (1276437113) |

## Decisions

| Need | Primary | Fallback | Notes |
|---|---|---|---|
| Web search | Tavily (if a key is set) | `ddgs`, no key | Both cached. ddgs is enough for development. |
| Official and pricing pages | `http.get` + trafilatura | Playwright for JavaScript pricing pages; manual URLs in `overrides.yaml` | robots.txt respected. |
| Hacker News complaints | Algolia HN Search API | — | Free, no key, generous limits. |
| Reddit complaints | Official API, **only if Reddit approves our request** | None: Reddit is left out | Reddit's Responsible Builder Policy (Reddit Help → Developer Platform & Data API Usage) requires approval before any API use, restricts "commercial and non-commercial mining", and says research using Reddit data outside its Researchers Program violates the policy. robots.txt also disallows fetching pages. So we don't fetch Reddit pages and don't mine Reddit search snippets. |
| GitHub issues (optional) | GitHub REST API | Skip products without public repos | Useful for open-source or developer-facing products. |
| App reviews (iOS) | Apple iTunes search API → customer-reviews RSS (JSON) | — | Up to 50 recent reviews per page, 4 pages per storefront. Each country's App Store has its own reviews, so we read the US, UK, Canada, Australia and India storefronts until 40 low-rated reviews are found. |
| App reviews (Android) | **Not used** | — | `google-play-scraper` fetches reviews from `play.google.com/_/PlayStoreUi/data/batchexecute`, and play.google.com's robots.txt has `Disallow: /_`. Only the app *pages* are allowed, and they don't contain the reviews. The extra Apple storefronts replace most of what Play gave us. |
| G2 / Capterra / Trustpilot | Search snippets only | Skip | Anti-bot measures and their terms forbid scraping. |

## Official pages: live check (2026-10-11)
`collect_official` on five demo competitors: Otter.ai, Fireflies.ai, Fathom and Granola returned their pricing page plus 3–5 feature, integration or security pages each. **tl;dv's pricing page is `/app/pricing`, which its robots.txt disallows**, so tl;dv pricing stays "unknown" unless a comparison article states it.

Optional browser rendering for JavaScript-only pages: `uv sync --extra browser && uv run playwright install chromium`. Without it, thin pages are kept and flagged with `meta["thin"]`.

## Complaints: live check (2026-10-13)
`collect_complaints` on seven demo competitors, limit 40 per source. App reviews are 1–3 stars
only, at least 40 characters.

| Competitor | HN comments | App reviews (Apple + Play) | Notes |
|---|---|---|---|
| Otter.ai | 39 | 40 (20 + 20) | |
| Fireflies.ai | 22 | 40 (20 + 20) | |
| Granola | 33 | 40 (21 + 19) | |
| Read.ai | 14 | 18 (7 + 11) | |
| Fellow | 5 | 24 (10 + 14) | The 5 HN comments are false positives ("Visiting Fellow") |
| tl;dv | 5 | 0 | No iOS app found; its Play app has no low-rated reviews |
| Fathom | 2 | 0 | Desktop product; its iOS app has almost no reviews |

Names that are ordinary words are the weak spot on Hacker News. `collectors/matching.py`
requires the product's domain, or the name close to a word from the market name, which removes
most unrelated comments but not all. Wrong app in the App Store? Set `apple_app_id` for the
company in `overrides.yaml`.

## Reddit status
A script app was created on 2026-10-11. Reddit's policy page says API use needs explicit approval, so `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` stay empty until an access request is approved. If you file one, describe the use accurately: non-commercial student project, read-only, low volume, posts analyzed by an LLM to find common complaints, not used for model training.

## Etiquette (enforced in `mia/http.py`)
- Descriptive User-Agent naming the project and repo.
- robots.txt respected for every web page. Only documented APIs skip the check.
- At least 1 second between requests to the same host; retries back off and honour Retry-After.
- Everything is cached, so each page is fetched once.
- Public pages only, never behind a login. Usernames are hashed before storing.
- Raw scraped data is never committed; the snapshot is shared privately.
