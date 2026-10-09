from __future__ import annotations

import httpx
import pytest

from mia import cache, runtime
from mia.http import FetchError, HttpClient, RobotsDisallowed


class Server:
    """Fake web server: maps (method, path) to a list of responses, served in order."""

    def __init__(self, routes: dict[tuple[str, str], list[httpx.Response]]):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key == ("GET", "/robots.txt") and key not in self.routes:
            return httpx.Response(404)
        queue = self.routes[key]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def count(self, path: str) -> int:
        return sum(r.url.path == path for r in self.requests)


@pytest.fixture
def clock():
    """Fake time: sleeping advances the clock, and every sleep is recorded."""

    class Clock:
        now = 0.0
        sleeps: list[float] = []

        def sleep(self, s: float) -> None:
            self.sleeps.append(s)
            self.now += s

        def __call__(self) -> float:
            return self.now

    return Clock()


@pytest.fixture
def make_client(settings, clock):
    runtime.configure(settings=settings)

    def make(routes) -> tuple[HttpClient, Server]:
        server = Server(routes)
        client = HttpClient(transport=httpx.MockTransport(server), sleep=clock.sleep, clock=clock)
        return client, server

    return make


URL = "https://site.example.com/page"


def test_get_returns_response_and_caches_it(make_client):
    client, server = make_client({("GET", "/page"): [httpx.Response(200, text="hello")]})
    first = client.get(URL)
    second = client.get(URL)
    assert first.ok and first.text == "hello" and first.status == 200
    assert second.text == "hello"
    assert server.count("/page") == 1


def test_404_is_cached_but_5xx_is_not(make_client):
    client, server = make_client(
        {
            ("GET", "/missing"): [httpx.Response(404)],
            ("GET", "/broken"): [httpx.Response(503)],
        }
    )
    client.get("https://site.example.com/missing")
    client.get("https://site.example.com/missing")
    assert server.count("/missing") == 1

    resp = client.get("https://site.example.com/broken")
    assert resp.status == 503 and not resp.ok
    assert server.count("/broken") == 4  # retried up to max_attempts
    client.get("https://site.example.com/broken")
    assert server.count("/broken") == 8  # not cached


def test_retries_with_backoff_then_succeeds(make_client, clock):
    client, server = make_client(
        {("GET", "/page"): [httpx.Response(429), httpx.Response(502), httpx.Response(200)]}
    )
    assert client.get(URL).ok
    assert [s for s in clock.sleeps if s >= 2] == [2.0, 4.0]


def test_retry_after_header_is_honoured(make_client, clock):
    client, _ = make_client(
        {("GET", "/page"): [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)]}
    )
    client.get(URL)
    assert 7.0 in clock.sleeps


def test_network_errors_raise_fetch_error_after_retries(settings, clock):
    runtime.configure(settings=settings)

    def fail(request):
        raise httpx.ConnectError("down")

    client = HttpClient(transport=httpx.MockTransport(fail), sleep=clock.sleep, clock=clock)
    with pytest.raises(FetchError):
        client.get(URL, respect_robots=False)


def test_requests_to_one_host_are_spaced_out(make_client, clock):
    client, _ = make_client({("GET", "/page"): [httpx.Response(200)]})
    client.get(URL, use_cache=False, respect_robots=False)
    client.get(URL, use_cache=False, respect_robots=False)
    assert clock.sleeps == [1.0]


def test_robots_txt_is_respected(make_client):
    client, server = make_client(
        {
            ("GET", "/robots.txt"): [httpx.Response(200, text="User-agent: *\nDisallow: /private")],
            ("GET", "/private/x"): [httpx.Response(200)],
            ("GET", "/public"): [httpx.Response(200)],
        }
    )
    assert client.get("https://site.example.com/public").ok
    with pytest.raises(RobotsDisallowed):
        client.get("https://site.example.com/private/x")
    assert server.count("/private/x") == 0
    assert server.count("/robots.txt") == 1  # fetched once per site


def test_robots_403_means_do_not_crawl(make_client):
    client, _ = make_client(
        {("GET", "/robots.txt"): [httpx.Response(403)], ("GET", "/page"): [httpx.Response(200)]}
    )
    with pytest.raises(RobotsDisallowed):
        client.get(URL)


def test_api_calls_can_skip_robots(make_client):
    client, server = make_client(
        {("GET", "/robots.txt"): [httpx.Response(403)], ("GET", "/page"): [httpx.Response(200)]}
    )
    assert client.get(URL, respect_robots=False).ok
    assert server.count("/robots.txt") == 0


def test_post_cache_key_ignores_headers(make_client):
    client, server = make_client({("POST", "/api"): [httpx.Response(200, json={"ok": True})]})
    api = "https://api.example.com/api"
    client.post_json(api, {"q": "x"}, headers={"Authorization": "Bearer one"})
    resp = client.post_json(api, {"q": "x"}, headers={"Authorization": "Bearer two"})
    assert resp.json() == {"ok": True}
    assert server.count("/api") == 1
    assert server.requests[0].headers["Authorization"] == "Bearer one"


def test_offline_mode_reads_cache_and_refuses_network(settings, make_client):
    client, server = make_client({("GET", "/page"): [httpx.Response(200, text="cached")]})
    client.get(URL)

    runtime.configure(settings=settings, offline=True)
    assert client.get(URL).text == "cached"
    with pytest.raises(cache.CacheMiss):
        client.get("https://site.example.com/never-fetched")
    assert server.count("/never-fetched") == 0


def test_user_agent_identifies_the_project(make_client):
    client, server = make_client({("GET", "/page"): [httpx.Response(200)]})
    client.get(URL)
    assert "MarketGapScout" in server.requests[-1].headers["User-Agent"]
