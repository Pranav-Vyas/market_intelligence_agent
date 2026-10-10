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
| Google Play pages | works | App page HTTP 200; robots.txt allows app pages |

## Decisions

| Need | Primary | Fallback | Notes |
|---|---|---|---|
| Web search | Tavily (if a key is set) | `ddgs`, no key | Both cached. ddgs is enough for development. |
| Official and pricing pages | `http.get` + trafilatura | Playwright for JavaScript pricing pages; manual URLs in `overrides.yaml` | robots.txt respected. |
| Hacker News complaints | Algolia HN Search API | — | Free, no key, generous limits. |
| Reddit complaints | Official API, **if Reddit approves an app** | Search results restricted to Reddit (`site:reddit.com <product> <problem>`), using the title and snippet only | **We do not fetch reddit.com pages or `.json` endpoints: their robots.txt disallows it.** Snippets are short, so Reddit will carry less weight than HN and the app stores. |
| App reviews (iOS) | Apple iTunes search API → customer-reviews RSS (JSON) | — | The RSS feed returns up to 50 recent reviews per page. |
| App reviews (Android) | `google-play-scraper` (added on Day 3) | Skip web-only products | App pages are allowed by robots.txt. |
| G2 / Capterra / Trustpilot | Search snippets only | Skip | Anti-bot measures and their terms forbid scraping. |

## Etiquette (enforced in `mia/http.py`)
- Descriptive User-Agent naming the project and repo.
- robots.txt respected for every web page. Only documented APIs skip the check.
- At least 1 second between requests to the same host; retries back off and honour Retry-After.
- Everything is cached, so each page is fetched once.
- Public pages only, never behind a login. Usernames are hashed before storing.
- Raw scraped data is never committed; the snapshot is shared privately.
