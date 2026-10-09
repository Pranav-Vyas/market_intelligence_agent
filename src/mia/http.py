"""Polite HTTP for every collector (task B1).

- Identifies itself with a descriptive User-Agent.
- Respects robots.txt for web pages (APIs are called with `respect_robots=False`).
- Waits at least 1 second between requests to the same host.
- Retries timeouts, 429 and 5xx with exponential backoff, honouring Retry-After.
- Caches successful responses (and 404/410) on disk; offline mode reads only from the cache.

Use the module-level `get()` and `post_json()`. Non-2xx responses are returned, not raised, so
check `.ok`. `FetchError` is raised only when the network itself keeps failing.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.robotparser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from mia import cache

log = logging.getLogger(__name__)

USER_AGENT = (
    "MarketGapScout/0.1 (student research project; "
    "+https://github.com/Pranav-Vyas/market_intelligence_agent)"
)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
CACHEABLE_STATUSES = frozenset({404, 410})
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0


class FetchError(Exception):
    pass


class RobotsDisallowed(FetchError):
    pass


@dataclass
class HttpResponse:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        return json.loads(self.text)


class HttpClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_interval_s: float = 1.0,
        timeout_s: float = 20.0,
        max_attempts: int = 4,
        user_agent: str = USER_AGENT,
    ):
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )
        self._user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._max_attempts = max_attempts
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ---- public API -------------------------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        respect_robots: bool = True,
        use_cache: bool = True,
    ) -> HttpResponse:
        def fetch() -> HttpResponse:
            if respect_robots and not self.allowed(url):
                raise RobotsDisallowed(f"robots.txt disallows {url}")
            return self._request("GET", url, params=params, headers=headers)

        return self._cached({"method": "GET", "url": url, "params": params}, fetch, use_cache)

    def post_json(
        self,
        url: str,
        payload: dict,
        *,
        headers: dict | None = None,
        use_cache: bool = True,
    ) -> HttpResponse:
        """POST to an API. Headers (which may hold keys) are not part of the cache key."""
        return self._cached(
            {"method": "POST", "url": url, "json": payload},
            lambda: self._request("POST", url, json_body=payload, headers=headers),
            use_cache,
        )

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                resp = self._request("GET", f"{origin}/robots.txt")
            except FetchError:
                parser.disallow_all = True  # can't read robots.txt: assume we may not crawl
            else:
                if resp.status in (401, 403):
                    parser.disallow_all = True
                elif resp.status >= 400:
                    parser.allow_all = True
                else:
                    parser.parse(resp.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(self._user_agent, url)

    # ---- internals --------------------------------------------------------------------------

    def _cached(self, key: dict, fetch: Callable[[], HttpResponse], use_cache: bool):
        if not use_cache:
            return fetch()
        stored: dict | None = None

        def fetch_cacheable() -> dict | None:
            nonlocal stored
            resp = fetch()
            stored = asdict(resp)
            return stored if resp.ok or resp.status in CACHEABLE_STATUSES else None

        value = cache.get_or_fetch("http", key, fetch_cacheable)
        return HttpResponse(**(value if value is not None else stored))

    def _request(self, method, url, *, params=None, json_body=None, headers=None) -> HttpResponse:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            self._throttle(url)
            try:
                r = self._client.request(
                    method, url, params=params, json=json_body, headers=headers
                )
            except httpx.TransportError as err:
                last_error = err
                log.info("%s %s failed (%s), attempt %d", method, url, err, attempt)
                wait = self._backoff(attempt)
            else:
                if r.status_code not in RETRY_STATUSES or attempt == self._max_attempts:
                    return HttpResponse(
                        url=url,
                        final_url=str(r.url),
                        status=r.status_code,
                        content_type=r.headers.get("content-type", ""),
                        text=r.text,
                    )
                log.info("%s %s returned %d, attempt %d", method, url, r.status_code, attempt)
                wait = self._retry_after(r) or self._backoff(attempt)
            if attempt < self._max_attempts:
                self._sleep(wait)
        raise FetchError(f"{method} {url} failed after {self._max_attempts} attempts: {last_error}")

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        last = self._last_request.get(host)
        if last is not None:
            wait = self._min_interval_s - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)

    @staticmethod
    def _retry_after(r: httpx.Response) -> float | None:
        value = r.headers.get("retry-after", "")
        return min(float(value), BACKOFF_MAX_S) if value.isdigit() else None


_default: HttpClient | None = None


def client() -> HttpClient:
    global _default
    if _default is None:
        _default = HttpClient()
    return _default


def get(url: str, **kwargs) -> HttpResponse:
    return client().get(url, **kwargs)


def post_json(url: str, payload: dict, **kwargs) -> HttpResponse:
    return client().post_json(url, payload, **kwargs)
