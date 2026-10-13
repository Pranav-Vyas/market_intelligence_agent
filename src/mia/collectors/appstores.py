"""Low-rated app reviews (task B6): Apple's customer-reviews RSS feed and Google Play.

Finding the right app matters more than fetching it: searching the App Store for "Fellow"
returns a coffee-kettle app first. For each store we take, in order:
  1. an id from `overrides.yaml` (`apple_app_id`, `google_play_id`),
  2. the store link on the competitor's own homepage,
  3. a search, accepted only if the app's developer site or its name and description match.

Google Play has no public API. Reviews come from `google-play-scraper`, wrapped here so its
calls are cached, throttled and skipped offline like every other request. Its `search()` is not
used, because robots.txt disallows /store/search; we use our own web search to find the app.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from itertools import zip_longest

from mia import cache, http
from mia.collectors.matching import author_hash, has_word, market_terms, plain_name, site
from mia.collectors.search import SearchError, search
from mia.discovery import load_overrides, name_key
from mia.rag.bm25 import tokenize
from mia.schemas import Competitor, Document, doc_id

log = logging.getLogger(__name__)

COUNTRY = "us"
MAX_RATING = 3  # 1-3 stars: these are the reviews that say what is wrong
MIN_TEXT_CHARS = 40
APPLE_SEARCH_URL = "https://itunes.apple.com/search"
APPLE_FEED_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}"
    "/sortBy=mostRecent/json"
)
APPLE_APP_URL = "https://apps.apple.com/{country}/app/id{app_id}"
APPLE_MAX_PAGES = 4  # 50 reviews per page
PLAY_APP_URL = "https://play.google.com/store/apps/details?id={app_id}"
PLAY_REVIEW_COUNT = 200  # one request
PLAY_CANDIDATES = 5  # search results checked against the app's own page, one request each

_APPLE_LINK = re.compile(r"apps\.apple\.com/[^\"'\s>]*?/id(\d+)")
_PLAY_LINK = re.compile(r"play\.google\.com/store/apps/details\?id=([A-Za-z0-9_.]+)")


def collect(
    c: Competitor, market: str, query: str | None = None, limit: int = 40
) -> list[Document]:
    want = None if query else limit  # with a query, scan everything and keep what matches
    apps = find_apps(c, market)
    apple = apple_reviews(c, apps["apple"], want) if apps["apple"] else []
    play = play_reviews(c, apps["google_play"], want) if apps["google_play"] else []
    docs = [d for pair in zip_longest(apple, play) for d in pair if d is not None]
    if query:
        docs = _matching(docs, query)
    return docs[:limit]


# ---- finding the app ----------------------------------------------------------------------------


def find_apps(c: Competitor, market: str) -> dict[str, str | None]:
    """App ids per store, or None where the product has no app (or none we can be sure of)."""
    override = next(
        (v for name, v in load_overrides(market).items() if name_key(name) == name_key(c.name)), {}
    )
    home = _homepage_html(c)
    apple = override.get("apple_app_id") or _first(_APPLE_LINK, home) or _apple_search(c, market)
    play = override.get("google_play_id") or _first(_PLAY_LINK, home) or _play_search(c, market)
    return {"apple": str(apple) if apple else None, "google_play": str(play) if play else None}


def _homepage_html(c: Competitor) -> str:
    try:
        resp = http.get(c.website)
    except (http.FetchError, cache.CacheMiss) as err:
        log.info("%s: homepage not available for store links: %s", c.name, err)
        return ""
    return resp.text if resp.ok else ""


def _first(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _is_this_product(c: Competitor, market: str, *, names: str, about: str, url: str) -> bool:
    if url and site(url) == site(c.website):
        return True
    plain = plain_name(c) or name_key(c.name)
    return has_word(names.lower(), plain) and any(
        term in about.lower() for term in market_terms(market)
    )


def _apple_search(c: Competitor, market: str) -> str | None:
    params = {"term": c.name, "entity": "software", "limit": 5, "country": COUNTRY}
    try:
        resp = http.get(APPLE_SEARCH_URL, params=params, respect_robots=False)  # documented API
    except (http.FetchError, cache.CacheMiss) as err:
        log.info("%s: App Store search failed: %s", c.name, err)
        return None
    for app in _json(resp).get("results", []):
        names = f"{app.get('trackName', '')} {app.get('sellerName', '')}"
        about = f"{app.get('trackName', '')} {app.get('description', '')}"
        if _is_this_product(c, market, names=names, about=about, url=app.get("sellerUrl") or ""):
            return str(app["trackId"])
    return None


def _play_search(c: Competitor, market: str) -> str | None:
    try:
        results = search(f"{c.name} app site:play.google.com", k=5)
    except (http.FetchError, cache.CacheMiss, SearchError) as err:
        log.info("%s: Google Play search failed: %s", c.name, err)
        return None
    ids = dict.fromkeys(m.group(1) for r in results if (m := _PLAY_LINK.search(r.url)))
    for app_id in list(ids)[:PLAY_CANDIDATES]:
        app = _play_call("app", app_id)
        if app is None:
            continue
        names = f"{app.get('title', '')} {app.get('developer', '')}"
        about = f"{app.get('title', '')} {app.get('description', '')}"
        if _is_this_product(c, market, names=names, about=about, url=app.get("website") or ""):
            return app_id
    return None


# ---- Apple --------------------------------------------------------------------------------------


def apple_reviews(c: Competitor, app_id: str, limit: int | None) -> list[Document]:
    url = APPLE_APP_URL.format(country=COUNTRY, app_id=app_id)
    docs: list[Document] = []
    for page in range(1, APPLE_MAX_PAGES + 1):
        feed_url = APPLE_FEED_URL.format(country=COUNTRY, page=page, app_id=app_id)
        try:
            resp = http.get(feed_url, respect_robots=False)  # documented feed
        except (http.FetchError, cache.CacheMiss) as err:
            log.info("%s: App Store reviews page %d failed: %s", c.name, page, err)
            break
        entries = _json(resp).get("feed", {}).get("entry", [])
        if isinstance(entries, dict):  # a single review comes back as an object, not a list
            entries = [entries]
        if not entries:
            break
        for e in entries:
            if "im:rating" not in e:
                continue
            title = _label(e, "title")
            doc = _review(
                c,
                url=url,
                review_id=_label(e, "id"),
                title=title,
                text=f"{title}\n{_label(e, 'content')}".strip(),
                rating=int(_label(e, "im:rating") or 0),
                author=e.get("author", {}).get("name", {}).get("label", ""),
                published_at=_parse_time(_label(e, "updated")),
                meta={"store": "apple", "app_id": app_id, "version": _label(e, "im:version")},
            )
            if doc is not None:
                docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
    return docs if limit is None else docs[:limit]


def _label(entry: dict, key: str) -> str:
    return (entry.get(key) or {}).get("label", "") or ""


# ---- Google Play --------------------------------------------------------------------------------


def play_reviews(c: Competitor, app_id: str, limit: int | None) -> list[Document]:
    url = PLAY_APP_URL.format(app_id=app_id)
    docs = []
    for r in _play_call("reviews", app_id) or []:
        doc = _review(
            c,
            url=url,
            review_id=r.get("id") or "",
            title="Google Play review",
            text=r.get("text") or "",
            rating=int(r.get("rating") or 0),
            author=r.get("author") or "",
            published_at=_parse_time(r.get("at") or ""),
            meta={"store": "google_play", "app_id": app_id, "version": r.get("version") or ""},
        )
        if doc is not None:
            docs.append(doc)
    return docs if limit is None else docs[:limit]


def _play_call(kind: str, app_id: str):
    """One cached, throttled call to Google Play: "app" → details dict, "reviews" → list of
    dicts. Returns None when it fails, when robots.txt disallows the app page, or offline."""
    url = PLAY_APP_URL.format(app_id=app_id)

    def fetch():
        if not http.client().allowed(url):
            log.info("robots.txt disallows %s", url)
            return None
        http.client().wait_turn(url)
        try:
            return _PLAY_FETCHERS[kind](app_id)
        except Exception as err:  # the library raises its own and urllib's errors
            log.warning("Google Play %s for %s failed: %s", kind, app_id, err)
            return None

    try:
        return cache.get_or_fetch("google_play", {"kind": kind, "app_id": app_id}, fetch)
    except cache.CacheMiss:
        return None


def _play_fetch_app(app_id: str) -> dict:
    from google_play_scraper import app

    info = app(app_id, lang="en", country=COUNTRY)
    return {
        "title": info.get("title") or "",
        "developer": info.get("developer") or "",
        "website": info.get("developerWebsite") or "",
        "description": info.get("description") or "",
    }


def _play_fetch_reviews(app_id: str) -> list[dict]:
    from google_play_scraper import Sort, reviews

    rows, _ = reviews(app_id, lang="en", country=COUNTRY, sort=Sort.NEWEST, count=PLAY_REVIEW_COUNT)
    return [
        {
            "id": r.get("reviewId"),
            "author": r.get("userName") or "",
            "text": r.get("content") or "",
            "rating": r.get("score"),
            "version": r.get("reviewCreatedVersion") or "",
            "at": r["at"].isoformat() if r.get("at") else "",
        }
        for r in rows
    ]


_PLAY_FETCHERS = {"app": _play_fetch_app, "reviews": _play_fetch_reviews}  # replaced in tests


# ---- shared -------------------------------------------------------------------------------------


def _review(
    c: Competitor,
    *,
    url: str,
    review_id: str,
    title: str,
    text: str,
    rating: int,
    author: str,
    published_at: datetime | None,
    meta: dict,
) -> Document | None:
    """A review as a Document, or None if it isn't a low-rated review with something to say."""
    if not review_id or not 1 <= rating <= MAX_RATING or len(text) < MIN_TEXT_CHARS:
        return None
    return Document(
        id=doc_id(url, review_id),  # every review of an app shares the app's URL
        url=url,
        source_type="app_store",
        company=c.name,
        title=title,
        text=text,
        published_at=published_at,
        meta={**meta, "rating": rating, "author_hash": author_hash(author)},
    )


def _json(resp: http.HttpResponse) -> dict:
    try:
        data = resp.json() if resp.ok else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _matching(docs: list[Document], query: str) -> list[Document]:
    """Reviews that share a word with the query, best overlap first."""
    terms = set(tokenize(query))
    scored = [(len(terms & set(tokenize(d.text))), d) for d in docs]
    scored = [(n, d) for n, d in scored if n > 0]
    return [d for _, d in sorted(scored, key=lambda s: s[0], reverse=True)]
