from __future__ import annotations

import re

import httpx
import pytest

from mia import http, runtime
from mia.collectors import appstores
from mia.collectors.complaints import collect_complaints
from mia.collectors.dedupe import dedupe, text_key
from mia.schemas import Competitor, Document, doc_id

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


FEEDS = {  # (country, page) -> entries
    ("us", 1): [
        {"im:name": {"label": "Acme Notes"}},  # older feeds start with the app itself
        entry("a1", 1, "Keeps crashing", f"Crashes when a meeting starts. {LONG}"),
        entry("a2", 5, "Love it", f"Best notes app I have used. {LONG}"),  # not a complaint
        entry("a3", 2, "Meh", "Bad."),  # too short
    ],
    ("us", 2): [entry("a4", 3, "Calendar sync", f"Duplicate calendar events every week. {LONG}")],
    ("gb", 1): [entry("g1", 1, "Battery", f"Battery drain is terrible. {LONG}", "John Doe")],
    ("ca", 1): entry("c1", 2, "Login", f"Logs me out every single day. {LONG}"),  # one review
}


@pytest.fixture
def stores(settings, monkeypatch):
    runtime.configure(settings=settings)
    state = {"home": HOME_WITH_LINKS, "itunes": [], "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        state["requests"].append(f"{host}{path}")
        if path == "/robots.txt":
            return httpx.Response(404)
        if host == "acme.example.com":
            return httpx.Response(200, html=state["home"])
        if path == "/search":
            return httpx.Response(200, json={"results": state["itunes"]})
        m = re.match(r"/(\w+)/rss/customerreviews/page=(\d+)/id=111/", path)
        entries = FEEDS.get((m.group(1), int(m.group(2))), []) if m else []
        return httpx.Response(200, json={"feed": {"entry": entries} if entries else {}})

    client = http.HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    monkeypatch.setattr(http, "_default", client)
    return state


def test_collects_low_rated_reviews_from_several_storefronts(stores):
    docs = collect_complaints(ACME, MARKET, "app_store")
    assert [(d.meta["country"], d.title) for d in docs] == [
        ("us", "Keeps crashing"),
        ("us", "Calendar sync"),
        ("gb", "Battery"),
        ("ca", "Login"),
    ]
    first = docs[0]
    assert first.source_type == "app_store"
    assert first.company == "Acme"
    assert first.url == "https://apps.apple.com/us/app/id111"
    assert first.id == doc_id(first.url, "a1")
    assert first.text.startswith("Keeps crashing\nCrashes when a meeting starts.")
    assert first.published_at is not None
    assert first.meta["rating"] == 1
    assert docs[2].url == "https://apps.apple.com/gb/app/id111"


def test_google_play_is_never_contacted(stores):
    collect_complaints(ACME, MARKET, "app_store")
    assert not any("play.google.com" in r for r in stores["requests"])


def test_usernames_are_hashed(stores):
    dumped = " ".join(d.model_dump_json() for d in collect_complaints(ACME, MARKET, "app_store"))
    assert "Jane Roe" not in dumped
    assert "John Doe" not in dumped


def test_limit_stops_reading_more_storefronts(stores):
    assert len(collect_complaints(ACME, MARKET, "app_store", limit=2)) == 2
    assert not any("/gb/" in r for r in stores["requests"])


def test_second_run_is_served_from_the_cache(stores):
    collect_complaints(ACME, MARKET, "app_store")
    n = len(stores["requests"])
    assert len(collect_complaints(ACME, MARKET, "app_store")) == 4
    assert len(stores["requests"]) == n


def test_query_keeps_only_matching_reviews(stores):
    docs = collect_complaints(ACME, MARKET, "app_store", query="calendar events")
    assert [d.title for d in docs] == ["Calendar sync"]
    # one shared word ("meeting") isn't enough for a two-word query
    assert collect_complaints(ACME, MARKET, "app_store", query="meeting dropped") == []
    assert len(collect_complaints(ACME, MARKET, "app_store", query="crashes")) == 1


def test_app_store_search_skips_other_products_with_the_same_name(stores):
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
    assert appstores.find_app(ACME, MARKET) == "2"


def test_app_store_search_accepts_an_app_from_the_same_developer_site(stores):
    stores["home"] = "<html><body>No store links here.</body></html>"
    stores["itunes"] = [
        {"trackId": 1, "trackName": "Acme", "sellerName": "Acme Kettles", "description": "Brew."},
        {"trackId": 7, "trackName": "Notes", "sellerUrl": "https://www.acme.example.com/app"},
    ]
    assert appstores.find_app(ACME, MARKET) == "7"


def test_override_wins(stores, monkeypatch):
    monkeypatch.setattr(appstores, "load_overrides", lambda market: {"acme": {"apple_app_id": 9}})
    assert appstores.find_app(ACME, MARKET) == "9"


def test_no_app_means_no_documents(stores):
    stores["home"] = "<html><body>No store links here.</body></html>"
    assert collect_complaints(ACME, MARKET, "app_store") == []


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
