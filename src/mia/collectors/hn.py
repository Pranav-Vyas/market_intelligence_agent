"""Hacker News comments that mention a competitor (task B5), from the Algolia HN Search API.

The API is free and needs no key. Algolia matches every word of the query, so we ask for the
name next to complaint words first ("otter.ai cancel"), then next to the market's own words
("otter meeting"), then the name alone, and stop as soon as there are enough comments. Each
comment must pass `matching.is_about`, because many product names are ordinary words.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import UTC, datetime

from mia import cache, http
from mia.collectors.matching import author_hash, is_about, market_terms, plain_name
from mia.schemas import Competitor, Document, doc_id

log = logging.getLogger(__name__)

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"
COMPLAINT_WORDS = ("problem", "issue", "cancel", "alternative")
HITS_PER_QUERY = 50
MIN_TEXT_CHARS = 80  # shorter comments rarely say what went wrong

_TAG = re.compile(r"<[^>]+>")


def collect(
    c: Competitor, market: str, query: str | None = None, limit: int = 40
) -> list[Document]:
    docs: dict[str, Document] = {}
    for q in queries(c, market, query):
        for hit in _search(q):
            doc = to_document(hit, c, q)
            if doc is None or doc.id in docs:
                continue
            if is_about(c, market, doc.text, doc.title):
                docs[doc.id] = doc
        if len(docs) >= limit:
            break
    return list(docs.values())[:limit]


def queries(c: Competitor, market: str, query: str | None = None) -> list[str]:
    name = c.name.strip()
    if query:
        return [f"{name} {query}"]
    short = plain_name(c) or name
    out = [f"{name} {word}" for word in COMPLAINT_WORDS]
    out += [f"{short} {term}" for term in market_terms(market)]
    out.append(name)
    return list(dict.fromkeys(out))


def to_document(hit: dict, c: Competitor, query: str) -> Document | None:
    text = clean(hit.get("comment_text") or "")
    if len(text) < MIN_TEXT_CHARS or not hit.get("objectID"):
        return None
    url = ITEM_URL.format(id=hit["objectID"])
    created = hit.get("created_at_i")
    return Document(
        id=doc_id(url),
        url=url,
        source_type="hn",
        company=c.name,
        title=hit.get("story_title") or "",
        text=text,
        published_at=datetime.fromtimestamp(created, UTC) if created else None,
        meta={
            "author_hash": author_hash(hit.get("author") or ""),
            "story_id": hit.get("story_id"),
            "query": query,
        },
    )


def clean(comment_html: str) -> str:
    """HN comment HTML → plain text with paragraph breaks."""
    text = _TAG.sub("", comment_html.replace("<p>", "\n\n"))
    lines = [" ".join(line.split()) for line in html.unescape(text).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _search(query: str) -> list[dict]:
    params = {
        "query": query,
        "tags": "comment",
        "hitsPerPage": HITS_PER_QUERY,
        "typoTolerance": "false",  # otherwise "otter" also matches "other"
    }
    try:
        resp = http.get(SEARCH_URL, params=params, respect_robots=False)  # documented API
    except (http.FetchError, cache.CacheMiss) as err:
        log.info("HN search %r failed: %s", query, err)
        return []
    if not resp.ok:
        log.info("HN search %r returned HTTP %d", query, resp.status)
        return []
    try:
        return resp.json().get("hits", [])
    except ValueError:
        return []
