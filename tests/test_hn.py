from __future__ import annotations

import httpx
import pytest

from mia import http, runtime
from mia.collectors import hn
from mia.collectors.complaints import collect_complaints
from mia.collectors.dedupe import dedupe
from mia.collectors.matching import is_about, market_terms
from mia.schemas import Competitor

MARKET = "AI meeting assistants"
OTTER = Competitor(name="Otter.ai", website="https://otter.ai")
FELLOW = Competitor(name="Fellow", website="https://fellow.app")
TLDV = Competitor(name="tl;dv", website="https://tldv.io")
READ = Competitor(name="Read.ai", website="https://www.read.ai")

LONG = "It kept happening for weeks and support never answered a single one of my emails."


def hit(object_id: str, text: str, title: str = "Ask HN: Which notetaker?", author: str = "pg"):
    return {
        "objectID": object_id,
        "comment_text": text,
        "story_title": title,
        "story_id": 1,
        "author": author,
        "created_at_i": 1_750_000_000,
    }


HITS = {
    "Otter.ai problem": [
        hit("1", f"I cancelled Otter.ai after it joined a call uninvited.<p>{LONG}"),
        hit("2", "Otter.ai is bad."),  # too short
        hit("3", f"An otter is a carnivorous mammal. {LONG}", title="Wildlife of the Pacific"),
    ],
    "Otter.ai issue": [
        hit("1", f"I cancelled Otter.ai after it joined a call uninvited.<p>{LONG}"),  # repeat
        hit("4", f"We moved off Otter because meeting summaries were wrong. {LONG}"),
    ],
    "Otter.ai calendar sync": [hit("9", f"otter.ai duplicates my calendar events. {LONG}")],
}


@pytest.fixture
def algolia(settings, monkeypatch):
    runtime.configure(settings=settings)
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "hn.algolia.com"
        assert request.url.params["typoTolerance"] == "false"
        queries.append(request.url.params["query"])
        return httpx.Response(200, json={"hits": HITS.get(request.url.params["query"], [])})

    client = http.HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    monkeypatch.setattr(http, "_default", client)
    return queries


def test_collects_relevant_comments_once(algolia):
    docs = collect_complaints(OTTER, MARKET, "hn")
    assert [d.url for d in docs] == [
        "https://news.ycombinator.com/item?id=1",
        "https://news.ycombinator.com/item?id=4",
    ]
    first = docs[0]
    assert first.source_type == "hn"
    assert first.company == "Otter.ai"
    assert first.title == "Ask HN: Which notetaker?"
    assert first.text.startswith(
        "I cancelled Otter.ai after it joined a call uninvited.\n\nIt kept"
    )
    assert first.published_at is not None


def test_usernames_are_hashed(algolia):
    doc = collect_complaints(OTTER, MARKET, "hn")[0]
    assert doc.meta["author_hash"] != "pg"
    assert "pg" not in doc.model_dump_json()


def test_a_comment_about_two_competitors_gets_one_document_per_competitor():
    text = f"We switched from Otter.ai to Read.ai for meeting notes. {LONG}"
    otter = hn.to_document(hit("7", text), OTTER, "q")
    read = hn.to_document(hit("7", text), READ, "q")
    assert otter.url == read.url
    assert otter.id != read.id
    assert len(dedupe([otter, read])) == 2


def test_queries_cover_complaint_words_market_words_and_the_name(algolia):
    collect_complaints(OTTER, MARKET, "hn")
    assert algolia == [
        "Otter.ai problem",
        "Otter.ai issue",
        "Otter.ai cancel",
        "Otter.ai alternative",
        "otter meeting",
        "otter assistant",
        "Otter.ai",
    ]


def test_stops_querying_once_the_limit_is_reached(algolia):
    docs = collect_complaints(OTTER, MARKET, "hn", limit=1)
    assert len(docs) == 1
    assert algolia == ["Otter.ai problem"]


def test_a_query_replaces_the_default_queries(algolia):
    docs = collect_complaints(OTTER, MARKET, "hn", query="calendar sync")
    assert algolia == ["Otter.ai calendar sync"]
    assert [d.url for d in docs] == ["https://news.ycombinator.com/item?id=9"]


def test_second_run_is_served_from_the_cache(algolia):
    collect_complaints(OTTER, MARKET, "hn")
    n = len(algolia)
    assert len(collect_complaints(OTTER, MARKET, "hn")) == 2
    assert len(algolia) == n


def test_failures_return_no_documents(settings, monkeypatch):
    runtime.configure(settings=settings)
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    client = http.HttpClient(transport=transport, sleep=lambda s: None, max_attempts=2)
    monkeypatch.setattr(http, "_default", client)
    assert collect_complaints(OTTER, MARKET, "hn") == []


def test_offline_without_a_cache_returns_no_documents(settings):
    runtime.configure(settings=settings, offline=True)
    assert collect_complaints(OTTER, MARKET, "hn") == []


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown source"):
        collect_complaints(OTTER, MARKET, "reddit")


def test_clean_turns_comment_html_into_text():
    raw = 'It&#x27;s <i>slow</i>.<p>See <a href="https://x.example">https:&#x2F;&#x2F;x.example</a>'
    assert hn.clean(raw) == "It's slow.\n\nSee https://x.example"


def test_market_terms():
    assert market_terms(MARKET) == ["meeting", "assistant"]
    assert market_terms("AI calendar / scheduling assistants") == [
        "calendar",
        "scheduling",
        "assistant",
    ]


@pytest.mark.parametrize(
    ("competitor", "text", "title", "expected"),
    [
        (OTTER, "I pay for otter.ai and regret it.", "", True),  # the domain is enough
        (OTTER, "We dropped Otter because the meeting notes were wrong.", "", True),
        (OTTER, "We saw an otter near the river.", "", False),
        (FELLOW, "We tried Fellow for meeting agendas and it lost our notes.", "", True),
        (FELLOW, "My fellow engineers hate every meeting.", "", False),
        (FELLOW, "Fellow engineers, skip this meeting.", "", False),  # sentence start
        (FELLOW, "He is a Research Fellow at MIT." + " x" * 200 + " meeting", "", False),  # far
        (FELLOW, "It lost my notes twice.", "Fellow: the AI meeting assistant", True),
        (TLDV, "TL;DV: the video is about compilers.", "", False),
        (TLDV, "tl;dv recorded our meeting without asking.", "", True),
        (READ, "I read the meeting notes.", "", False),
        (READ, "Read AI joins every meeting on its own.", "", True),
    ],
)
def test_is_about(competitor, text, title, expected):
    assert is_about(competitor, MARKET, text, title) is expected
