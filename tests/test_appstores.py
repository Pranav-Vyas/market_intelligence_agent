from __future__ import annotations

import httpx
import pytest

from mia import http, runtime
from mia.collectors import appstores
from mia.collectors.complaints import collect_complaints
from mia.collectors.dedupe import dedupe, text_key
from mia.schemas import Competitor, Document, SearchResult, doc_id

MARKET = "AI meeting assistants"
ACME = Competitor(name="Acme", website="https://acme.example.com")
LONG = "It has been like this for three releases now and nobody replies to support tickets."

HOME_WITH_LINKS = """<html><body>
  <a href="https://apps.apple.com/us/app/acme-notes/id111?mt=8">App Store</a>
  <a href="https://play.google.com/store/apps/details?id=com.acme.notes&hl=en">Google Play</a>
</body></html>"""


def entry(review_id: str, rating: int, title: str, content: str, author: str = "Jane Roe"):
    return {
        "id": {"label": review_id},
        "title": {"label": title},
        "content": {"label": content},
        "im:rating": {"label": str(rating)},
        "im:version": {"label": "2.1"},
        "author": {"name": {"label": author}},
        "updated": {"label": "2026-09-30T10:00:00-07:00"},
    }


APPLE_PAGES = {
    1: [
        entry("a1", 1, "Keeps crashing", f"Crashes when a meeting starts. {LONG}"),
        entry("a2", 5, "Love it", f"Best notes app I have used. {LONG}"),  # not a complaint
        entry("a3", 2, "Meh", "Bad."),  # too short
    ],
    2: [entry("a4", 3, "Calendar sync", f"Duplicate calendar events every week. {LONG}")],
}

PLAY_REVIEWS = [
    {"id": "p1", "author": "John Doe", "text": f"Battery drain is terrible. {LONG}", "rating": 1},
    {"id": "p2", "author": "Ann", "text": f"Works great for me every day. {LONG}", "rating": 4},
    {  # the same review as Apple's a1, posted to both stores
        "id": "p3",
        "author": "Bo",
        "text": f"Keeps crashing. Crashes when a meeting starts. {LONG}",
        "rating": 2,
    },
]


@pytest.fixture
def stores(settings, monkeypatch):
    runtime.configure(settings=settings)
    state = {
        "home": HOME_WITH_LINKS,
        "itunes": [],
        "requests": [],
        "play_calls": [],
        "play_apps": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        state["requests"].append(f"{host}{path}")
        if path == "/robots.txt":
            return httpx.Response(404)
        if host == "acme.example.com":
            return httpx.Response(200, html=state["home"])
        if path == "/search":
            return httpx.Response(200, json={"results": state["itunes"]})
        for page, entries in APPLE_PAGES.items():
            if f"/page={page}/" in path:
                return httpx.Response(200, json={"feed": {"entry": entries}})
        return httpx.Response(200, json={"feed": {}})

    client = http.HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    monkeypatch.setattr(http, "_default", client)

    def play_reviews(app_id: str) -> list[dict]:
        state["play_calls"].append(("reviews", app_id))
        return [{**r, "version": "9", "at": "2026-09-29T08:00:00"} for r in PLAY_REVIEWS]

    def play_app(app_id: str) -> dict:
        state["play_calls"].append(("app", app_id))
        return state["play_apps"][app_id]

    monkeypatch.setattr(appstores, "_PLAY_FETCHERS", {"app": play_app, "reviews": play_reviews})
    monkeypatch.setattr(appstores, "search", lambda query, k=5: [])
    return state


def test_collects_low_rated_reviews_from_both_stores(stores):
    docs = collect_complaints(ACME, MARKET, "app_store")
    assert [(d.meta["store"], d.meta["rating"]) for d in docs] == [
        ("apple", 1),
        ("google_play", 1),
        ("apple", 3),
        # the second Play review repeats an Apple review word for word, so it is dropped
    ]
    apple = docs[0]
    assert apple.source_type == "app_store"
    assert apple.company == "Acme"
    assert apple.url == "https://apps.apple.com/us/app/id111"
    assert apple.id == doc_id(apple.url, "a1")
    assert apple.title == "Keeps crashing"
    assert apple.text.startswith("Keeps crashing\nCrashes when a meeting starts.")
    assert apple.published_at is not None
    assert docs[1].url == "https://play.google.com/store/apps/details?id=com.acme.notes"
    assert len({d.id for d in docs}) == 3


def test_usernames_are_hashed(stores):
    dumped = " ".join(d.model_dump_json() for d in collect_complaints(ACME, MARKET, "app_store"))
    assert "Jane Roe" not in dumped
    assert "John Doe" not in dumped


def test_limit_and_cache(stores):
    assert len(collect_complaints(ACME, MARKET, "app_store", limit=2)) == 2
    requests, play_calls = len(stores["requests"]), len(stores["play_calls"])
    assert len(collect_complaints(ACME, MARKET, "app_store", limit=2)) == 2
    assert len(stores["requests"]) == requests
    assert len(stores["play_calls"]) == play_calls


def test_query_keeps_only_matching_reviews(stores):
    docs = collect_complaints(ACME, MARKET, "app_store", query="calendar events")
    assert [d.title for d in docs] == ["Calendar sync"]


def test_apple_search_skips_other_products_with_the_same_name(stores):
    stores["home"] = "<html><body>No store links here.</body></html>"
    stores["itunes"] = [
        {"trackId": 1, "trackName": "Acme", "sellerName": "Acme Kettles", "description": "Brew."},
        {
            "trackId": 2,
            "trackName": "Acme - AI Meeting Assistant",
            "sellerName": "Acme Inc.",
            "description": "Notes for every meeting.",
        },
    ]
    assert appstores.find_apps(ACME, MARKET) == {"apple": "2", "google_play": None}


def test_apple_search_accepts_an_app_from_the_same_developer_site(stores):
    stores["home"] = "<html><body>No store links here.</body></html>"
    stores["itunes"] = [
        {"trackId": 1, "trackName": "Acme", "sellerName": "Acme Kettles", "description": "Brew."},
        {"trackId": 7, "trackName": "Notes", "sellerUrl": "https://www.acme.example.com/app"},
    ]
    assert appstores.find_apps(ACME, MARKET)["apple"] == "7"


def test_google_play_app_is_found_by_web_search_and_checked(stores, monkeypatch):
    stores["home"] = "<html><body>No store links here.</body></html>"
    stores["play_apps"] = {
        "com.kettles.acme": {"title": "Acme", "developer": "Acme Kettles", "description": "Brew."},
        "com.acme.notes": {
            "title": "Acme Notes",
            "developer": "Acme Inc.",
            "website": "https://www.acme.example.com/",
            "description": "Notes.",
        },
    }
    play = "https://play.google.com/store/apps/details?id="
    results = [
        SearchResult(url=f"{play}com.kettles.acme", title="Acme", snippet=""),
        SearchResult(url=f"{play}com.acme.notes&hl=en", title="Acme Notes", snippet=""),
    ]
    monkeypatch.setattr(appstores, "search", lambda query, k=5: results)
    assert appstores.find_apps(ACME, MARKET)["google_play"] == "com.acme.notes"
    # google-play-scraper's own search is never used: robots.txt disallows /store/search
    assert all("store/search" not in r for r in stores["requests"])


def test_no_app_means_no_documents(stores):
    stores["home"] = "<html><body>No store links here.</body></html>"
    assert collect_complaints(ACME, MARKET, "app_store") == []


def test_google_play_failure_is_not_fatal(stores, monkeypatch):
    def broken(app_id: str):
        raise RuntimeError("layout changed")

    monkeypatch.setattr(appstores, "_PLAY_FETCHERS", {"app": broken, "reviews": broken})
    docs = collect_complaints(ACME, MARKET, "app_store")
    assert {d.meta["store"] for d in docs} == {"apple"}


def test_offline_without_a_cache_returns_no_documents(settings):
    runtime.configure(settings=settings, offline=True)
    assert collect_complaints(ACME, MARKET, "app_store") == []


def _doc(url: str, text: str, suffix: str = "") -> Document:
    return Document(id=doc_id(url, suffix), url=url, source_type="hn", company="Acme", text=text)


def test_dedupe_drops_repeated_ids_and_texts_and_keeps_the_first():
    docs = [
        _doc("https://a.example/1", "It crashes all the time."),
        _doc("https://a.example/1", "Edited: it crashes sometimes."),  # same id
        _doc("https://a.example/2", "it  CRASHES all the time"),  # same text
        _doc("https://a.example/app", "Too expensive.", "r1"),
        _doc("https://a.example/app", "Sync is broken.", "r2"),  # same URL, another review
    ]
    assert [d.text for d in dedupe(docs)] == [
        "It crashes all the time.",
        "Too expensive.",
        "Sync is broken.",
    ]


def test_dedupe_is_per_company():
    acme = _doc("https://a.example/1", "Otter and Acme both crash.")
    other = acme.model_copy(update={"company": "Otter"})
    assert dedupe([acme, other, acme]) == [acme, other]


def test_text_key_ignores_case_punctuation_and_spacing():
    assert text_key("Too  expensive!") == text_key("too expensive")
    assert text_key("too expensive") != text_key("too cheap")
