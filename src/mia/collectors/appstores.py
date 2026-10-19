"""Low-rated App Store reviews (task B6), from Apple's customer-reviews RSS feed.

Finding the right app matters more than fetching it: searching the App Store for "Fellow"
returns a coffee-kettle app first. We take, in order:
  1. an `apple_app_id` from `overrides.yaml`,
  2. the App Store link on the competitor's own homepage,
  3. an App Store search, accepted only if the app's developer site or its name and description
     match.

Each country's App Store has its own reviews, so we read several English-language storefronts
for the same app. Google Play is not used: its reviews come from `/_/PlayStoreUi/...`, which
play.google.com's robots.txt disallows (docs/data_sources.md).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from mia import cache, http
from mia.collectors.matching import author_hash, has_word, market_terms, plain_name, site
from mia.discovery import load_overrides, name_key
from mia.rag.bm25 import tokenize
from mia.schemas import Competitor, Document, doc_id

log = logging.getLogger(__name__)

COUNTRIES = ("us", "gb", "ca", "au", "in")  # English-language storefronts, most reviews first
MAX_RATING = 3  # 1-3 stars: these are the reviews that say what is wrong
MIN_TEXT_CHARS = 40
SEARCH_URL = "https://itunes.apple.com/search"
FEED_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}"
    "/sortBy=mostRecent/json"
)
APP_URL = "https://apps.apple.com/{country}/app/id{app_id}"
MAX_PAGES = 4  # per storefront, 50 reviews per page

_APP_LINK = re.compile(r"apps\.apple\.com/[^\"'\s>]*?/id(\d+)")


def collect(
    c: Competitor, market: str, query: str | None = None, limit: int = 40
) -> list[Document]:
    app_id = find_app(c, market)
    if app_id is None:
        return []
    want = None if query else limit  # with a query, scan everything and keep what matches
    docs: list[Document] = []
    for country in COUNTRIES:
        docs += reviews(c, app_id, country, None if want is None else want - len(docs))
        if want is not None and len(docs) >= want:
            break
    if query:
        docs = _matching(docs, query)
    return docs[:limit]


# ---- finding the app ----------------------------------------------------------------------------


def find_app(c: Competitor, market: str) -> str | None:
    """The competitor's App Store app id, or None when it has no app (or none we can be sure of)."""
    override = next(
        (v for name, v in load_overrides(market).items() if name_key(name) == name_key(c.name)), {}
    )
    app_id = override.get("apple_app_id") or _first(_APP_LINK, _homepage_html(c))
    app_id = app_id or _search(c, market)
    return str(app_id) if app_id else None


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


def _search(c: Competitor, market: str) -> str | None:
    params = {"term": c.name, "entity": "software", "limit": 5, "country": COUNTRIES[0]}
    try:
        resp = http.get(SEARCH_URL, params=params, respect_robots=False)  # documented API
    except (http.FetchError, cache.CacheMiss) as err:
        log.info("%s: App Store search failed: %s", c.name, err)
        return None
    for app in _json(resp).get("results", []):
        names = f"{app.get('trackName', '')} {app.get('sellerName', '')}"
        about = f"{app.get('trackName', '')} {app.get('description', '')}"
        if _is_this_product(c, market, names=names, about=about, url=app.get("sellerUrl") or ""):
            return str(app["trackId"])
    return None


# ---- reviews ------------------------------------------------------------------------------------


def reviews(c: Competitor, app_id: str, country: str, limit: int | None) -> list[Document]:
    """Low-rated reviews from one storefront, newest first."""
    url = APP_URL.format(country=country, app_id=app_id)
    docs: list[Document] = []
    for page in range(1, MAX_PAGES + 1):
        feed_url = FEED_URL.format(country=country, page=page, app_id=app_id)
        try:
            resp = http.get(feed_url, respect_robots=False)  # documented feed
        except (http.FetchError, cache.CacheMiss) as err:
            log.info("%s: App Store (%s) reviews page %d failed: %s", c.name, country, page, err)
            break
        entries = _json(resp).get("feed", {}).get("entry", [])
        if isinstance(entries, dict):  # a single review comes back as an object, not a list
            entries = [entries]
        if not entries:
            break
        for e in entries:
            doc = _review(c, url, e, meta={"store": "apple", "country": country, "app_id": app_id})
            if doc is not None:
                docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
    return docs if limit is None else docs[:limit]


def _review(c: Competitor, url: str, e: dict, meta: dict) -> Document | None:
    """A feed entry as a Document, or None if it isn't a low-rated review with something to say."""
    if "im:rating" not in e:  # the first entry of older feeds describes the app itself
        return None
    review_id, title = _label(e, "id"), _label(e, "title")
    text = f"{title}\n{_label(e, 'content')}".strip()
    rating = int(_label(e, "im:rating") or 0)
    if not review_id or not 1 <= rating <= MAX_RATING or len(text) < MIN_TEXT_CHARS:
        return None
    author = e.get("author", {}).get("name", {}).get("label", "")
    return Document(
        id=doc_id(url, review_id),  # every review of an app shares the app's URL
        url=url,
        source_type="app_store",
        company=c.name,
        title=title,
        text=text,
        published_at=_parse_time(_label(e, "updated")),
        meta={
            **meta,
            "version": _label(e, "im:version"),
            "rating": rating,
            "author_hash": author_hash(author),
        },
    )


def _label(entry: dict, key: str) -> str:
    return (entry.get(key) or {}).get("label", "") or ""


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
    """Reviews that share at least two of the query's words (one for a one-word query), best
    overlap first. One shared word is too loose: almost every review says "recording"."""
    terms = set(tokenize(query))
    needed = min(2, len(terms))
    scored = [(len(terms & set(tokenize(d.text))), d) for d in docs]
    scored = [(n, d) for n, d in scored if n >= needed and n > 0]
    return [d for _, d in sorted(scored, key=lambda s: s[0], reverse=True)]
