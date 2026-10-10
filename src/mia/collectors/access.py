"""Data-source access check (task B3): `uv run mia check-access`.

Hits each source once, bypassing the cache, and reports works / blocked / missing key. Re-run it
whenever a collector starts failing; sites and API policies change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mia import http, runtime

PROBE = "otter.ai"


@dataclass
class AccessResult:
    source: str
    status: str  # "works" | "blocked" | "no key" | "error"
    detail: str


def check_all() -> list[AccessResult]:
    checks: list[tuple[str, Callable[[], AccessResult]]] = [
        ("Gemini API", _gemini),
        ("Tavily search", _tavily),
        ("ddgs search (no key)", _ddgs),
        ("Hacker News (Algolia API)", _hn),
        ("Reddit official API", _reddit_api),
        ("Reddit public JSON", _reddit_json),
        ("Apple App Store reviews (RSS)", _apple),
        ("Google Play pages", _google_play),
    ]
    results = []
    for name, check in checks:
        try:
            result = check()
        except http.RobotsDisallowed as err:
            result = AccessResult(name, "blocked", f"robots.txt disallows it ({err})")
        except Exception as err:  # report every failure, never crash the whole check
            result = AccessResult(name, "error", f"{type(err).__name__}: {err}")
        result.source = name
        results.append(result)
    return results


def _gemini() -> AccessResult:
    key = runtime.settings().gemini_api_key
    if not key:
        return AccessResult("", "no key", "Set GEMINI_API_KEY in .env (needed from Day 2)")
    from mia.llm import GeminiBackend

    models = GeminiBackend(key).list_models()
    return AccessResult("", "works", f"{len(models)} models available; see `mia models`")


def _tavily() -> AccessResult:
    key = runtime.settings().tavily_api_key
    if not key:
        return AccessResult("", "no key", "Optional: set TAVILY_API_KEY; ddgs is used without it")
    resp = http.post_json(
        "https://api.tavily.com/search",
        {"query": f"{PROBE} alternatives", "max_results": 3},
        headers={"Authorization": f"Bearer {key}"},
        use_cache=False,
    )
    if not resp.ok:
        return AccessResult("", "error", f"HTTP {resp.status}: {resp.text[:120]}")
    return AccessResult("", "works", f"{len(resp.json().get('results', []))} results")


def _ddgs() -> AccessResult:
    from ddgs import DDGS

    rows = DDGS().text(f"{PROBE} alternatives", max_results=3)
    status = "works" if rows else "error"
    return AccessResult("", status, f"{len(rows)} results")


def _hn() -> AccessResult:
    resp = http.get(
        "https://hn.algolia.com/api/v1/search",
        params={"query": PROBE, "tags": "comment", "hitsPerPage": 3},
        respect_robots=False,
        use_cache=False,
    )
    if not resp.ok:
        return AccessResult("", "error", f"HTTP {resp.status}")
    return AccessResult("", "works", f"{resp.json().get('nbHits', 0)} matching comments")


def _reddit_api() -> AccessResult:
    s = runtime.settings()
    if not (s.reddit_client_id and s.reddit_client_secret):
        return AccessResult(
            "",
            "no key",
            "Register a script app at reddit.com/prefs/apps; new apps may need Reddit's approval",
        )
    return AccessResult("", "works", "Credentials set (verified when the Reddit collector runs)")


def _reddit_json() -> AccessResult:
    resp = http.get(
        "https://www.reddit.com/search.json",
        params={"q": PROBE, "limit": 3},
        use_cache=False,
    )
    if not resp.ok:
        return AccessResult("", "blocked", f"HTTP {resp.status}")
    return AccessResult("", "works", f"{len(resp.json()['data']['children'])} posts")


def _apple() -> AccessResult:
    lookup = http.get(
        "https://itunes.apple.com/search",
        params={"term": PROBE, "entity": "software", "limit": 1},
        respect_robots=False,
        use_cache=False,
    )
    apps = lookup.json().get("results", []) if lookup.ok else []
    if not apps:
        return AccessResult("", "error", f"App lookup failed (HTTP {lookup.status})")
    app_id, app_name = apps[0]["trackId"], apps[0].get("trackName", "")
    feed = http.get(
        f"https://itunes.apple.com/us/rss/customerreviews/id={app_id}/sortBy=mostRecent/json",
        respect_robots=False,
        use_cache=False,
    )
    if not feed.ok:
        return AccessResult("", "error", f"Review feed HTTP {feed.status}")
    entries = feed.json().get("feed", {}).get("entry", [])
    return AccessResult("", "works", f"{len(entries)} recent reviews for {app_name} ({app_id})")


def _google_play() -> AccessResult:
    url = "https://play.google.com/store/apps/details?id=com.google.android.apps.docs&hl=en"
    resp = http.get(url, use_cache=False)
    status = "works" if resp.ok else "error"
    return AccessResult("", status, f"App page HTTP {resp.status}; robots.txt allows app pages")


def as_markdown(results: list[AccessResult]) -> str:
    lines = ["| Source | Status | Detail |", "|---|---|---|"]
    lines += [f"| {r.source} | {r.status} | {r.detail.replace('|', '/')} |" for r in results]
    return "\n".join(lines)
