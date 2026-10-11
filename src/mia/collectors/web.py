"""Official pages (task B4): homepage, feature/integration/security pages and pricing, cleaned to
Markdown text with trafilatura.

Pages that come back nearly empty are usually rendered with JavaScript. For those we try a
headless browser, which is optional:
    uv sync --extra browser && uv run playwright install chromium
Without it, the short text is kept and the page is marked `meta["thin"] = True`.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import trafilatura

from mia import cache, http
from mia.schemas import Competitor, Document, SourceType, doc_id

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 400  # below this, the page was probably rendered with JavaScript
MAX_FEATURE_PAGES = 4  # per competitor, on top of the homepage and the pricing page
MAX_PER_SECTION = 2  # e.g. at most 2 pages under /integrations/, so one section can't crowd out

# Path or link-text keywords, in priority order.
PRICING_WORDS = ("pricing", "plans", "price")
FEATURE_WORDS = (
    "features",
    "feature",
    "integrations",
    "integration",
    "product",
    "security",
    "how-it-works",
    "use-cases",
    "solutions",
)
SKIP_WORDS = (
    "blog",
    "login",
    "log-in",
    "signin",
    "sign-in",
    "signup",
    "sign-up",
    "register",
    "careers",
    "jobs",
    "privacy",
    "terms",
    "legal",
    "cookie",
    "press",
    "status",
    "download",
    "contact",
)


# ---- public API ------------------------------------------------------------------------------


def fetch_page(url: str, source_type: SourceType, company: str | None) -> Document | None:
    doc, _ = _fetch(url, source_type, company)
    return doc


def collect_official(c: Competitor) -> list[Document]:
    home, html = _fetch(c.website, "official", c.name)
    if home is None:
        log.warning("%s: homepage %s could not be fetched", c.name, c.website)
        return []

    links = find_links(html, home.url)
    pricing_url = c.pricing_url or (links["pricing"][0] if links["pricing"] else None)
    pricing_url = pricing_url or urljoin(home.url, "/pricing")

    docs = [home]
    pricing = fetch_page(pricing_url, "pricing", c.name)
    if pricing is not None:
        docs.append(pricing)
    for url in links["features"][:MAX_FEATURE_PAGES]:
        page = fetch_page(url, "official", c.name)
        if page is not None:
            docs.append(page)

    unique = {d.id: d for d in docs}  # redirects can make two links land on the same page
    return list(unique.values())


# ---- fetching and cleaning ---------------------------------------------------------------------


def _fetch(url: str, source_type: SourceType, company: str | None) -> tuple[Document | None, str]:
    try:
        resp = http.get(url)
    except http.RobotsDisallowed:
        log.info("robots.txt disallows %s", url)
        return None, ""
    except (http.FetchError, cache.CacheMiss) as err:
        log.info("could not fetch %s: %s", url, err)
        return None, ""
    if not resp.ok or "html" not in resp.content_type.lower():
        return None, ""

    html = resp.text
    title, text = extract_text(html)
    meta: dict = {}
    if len(text) < MIN_TEXT_CHARS:
        rendered = _renderer(resp.final_url)
        if rendered:
            r_title, r_text = extract_text(rendered)
            if len(r_text) > len(text):
                html, title, text = rendered, r_title or title, r_text
                meta["rendered"] = True
        if len(text) < MIN_TEXT_CHARS:
            meta["thin"] = True
    if not text:
        return None, html

    final = _canonical(resp.final_url)
    doc = Document(
        id=doc_id(final),
        url=final,
        source_type=source_type,
        company=company,
        title=title,
        text=text,
        meta=meta,
    )
    return doc, html


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, main text as Markdown). Falls back to plain page text when trafilatura
    finds no main content, which happens on pricing pages built from cards and tables."""
    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        include_comments=False,
        favor_recall=True,
    )
    if not text or len(text) < MIN_TEXT_CHARS:
        plain = trafilatura.html2txt(html) or ""
        if len(plain) > len(text or ""):
            text = plain
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    return title.strip(), (text or "").strip()


# ---- links -----------------------------------------------------------------------------------


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def find_links(html: str, base_url: str) -> dict[str, list[str]]:
    """Same-site links that look like pricing or feature pages, best match first."""
    parser = _LinkParser()
    parser.feed(html)
    home = _site(base_url)
    scored: dict[str, dict[str, int]] = {"pricing": {}, "features": {}}
    for href, text in parser.links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        url = _canonical(urljoin(base_url, href))
        if _site(url) != home or url.rstrip("/") == _canonical(base_url).rstrip("/"):
            continue
        words = f"{urlsplit(url).path.lower()} {text.lower()}"
        if any(w in words for w in SKIP_WORDS):
            continue
        for kind, keywords in (("pricing", PRICING_WORDS), ("features", FEATURE_WORDS)):
            rank = next((i for i, w in enumerate(keywords) if w in words), None)
            if rank is not None:
                best = scored[kind].get(url)
                scored[kind][url] = rank if best is None else min(best, rank)
                break
    return {kind: _best_first(urls) for kind, urls in scored.items()}


def _best_first(scored: dict[str, int]) -> list[str]:
    """Best keyword match first, then shallower paths; at most MAX_PER_SECTION per section."""

    def depth(url: str) -> int:
        return len([p for p in urlsplit(url).path.split("/") if p])

    ranked = sorted(scored, key=lambda u: (scored[u], depth(u)))
    per_section: dict[str, int] = {}
    out = []
    for url in ranked:
        section = next((p for p in urlsplit(url).path.split("/") if p), "")
        if per_section.get(section, 0) < MAX_PER_SECTION:
            per_section[section] = per_section.get(section, 0) + 1
            out.append(url)
    return out


def _canonical(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", "", ""))


def _site(url: str) -> str:
    """Registrable-ish site name: host without 'www.', so docs.x.com and x.com match."""
    host = urlsplit(url).netloc.lower().removeprefix("www.")
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# ---- optional browser rendering -------------------------------------------------------------


def render_with_browser(url: str) -> str | None:
    """Rendered HTML of a JavaScript page, or None if Playwright isn't available."""
    if not http.client().allowed(url):
        return None

    def render() -> str | None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            log.info("Playwright not installed; keeping the unrendered text of %s", url)
            return None
        http.client().wait_turn(url)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(user_agent=http.USER_AGENT)
                page.goto(url, wait_until="networkidle", timeout=30_000)
                html = page.content()
                browser.close()
                return html
        except Exception as err:  # missing browser binary, timeouts, crashes
            log.warning("browser rendering failed for %s: %s", url, err)
            return None

    try:
        return cache.get_or_fetch("rendered", url, render)
    except cache.CacheMiss:
        return None


_renderer = render_with_browser  # replaced in tests
