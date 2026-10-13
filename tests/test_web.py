from __future__ import annotations

import httpx
import pytest

from mia import http, runtime
from mia.collectors import web
from mia.schemas import Competitor

FILLER = " ".join(["Meetings are recorded, transcribed and summarized automatically."] * 12)


def page(title: str, body: str) -> str:
    head = f"<head><title>{title}</title></head>"
    return f"<html>{head}<body><main><h1>{title}</h1>{body}</main></body></html>"


HOME = page(
    "Acme Notes",
    f"""<nav>
      <a href="/features">Features</a>
      <a href="/integrations">Integrations</a>
      <a href="https://acme.example.com/pricing#plans">Pricing</a>
      <a href="/blog/launch">Blog</a>
      <a href="/login">Log in</a>
      <a href="https://other.example.org/features">Partner</a>
      <a href="mailto:hi@acme.example.com">Email</a>
    </nav><p>{FILLER}</p>""",
)

SITE = {
    "/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /private"),
    "/": httpx.Response(200, html=HOME),
    "/features": httpx.Response(200, html=page("Features", f"<p>Speaker labels. {FILLER}</p>")),
    "/integrations": httpx.Response(200, html=page("Integrations", f"<p>Slack. {FILLER}</p>")),
    "/pricing": httpx.Response(200, html=page("Pricing", "<div id='app'></div>")),  # JS-rendered
    "/plans": httpx.Response(200, html=page("Plans", f"<p>Pro is $12 per seat. {FILLER}</p>")),
    "/file.pdf": httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
    "/private/x": httpx.Response(200, html=page("Secret", f"<p>{FILLER}</p>")),
}


@pytest.fixture
def site(settings, monkeypatch):
    runtime.configure(settings=settings)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return SITE.get(request.url.path, httpx.Response(404))

    client = http.HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    monkeypatch.setattr(http, "_default", client)
    rendered: list[str] = []

    def fake_renderer(url: str) -> str | None:
        rendered.append(url)
        return page("Pricing", f"<p>Free plan. Pro costs $12 per seat per month. {FILLER}</p>")

    monkeypatch.setattr(web, "_renderer", fake_renderer)
    return {"requested": requested, "rendered": rendered}


ACME = Competitor(name="Acme", website="https://acme.example.com")


def test_collects_homepage_pricing_and_feature_pages(site):
    docs = web.collect_official(ACME)
    by_url = {d.url: d for d in docs}
    assert set(by_url) == {
        "https://acme.example.com/",
        "https://acme.example.com/pricing",
        "https://acme.example.com/features",
        "https://acme.example.com/integrations",
    }
    assert by_url["https://acme.example.com/pricing"].source_type == "pricing"
    assert by_url["https://acme.example.com/features"].source_type == "official"
    assert {d.company for d in docs} == {"Acme"}
    assert "/blog/launch" not in site["requested"] and "/login" not in site["requested"]


def test_js_pricing_page_falls_back_to_browser_rendering(site):
    pricing = next(d for d in web.collect_official(ACME) if d.source_type == "pricing")
    assert site["rendered"] == ["https://acme.example.com/pricing"]
    assert "$12 per seat" in pricing.text
    assert pricing.meta == {"rendered": True}


def test_thin_page_is_kept_and_flagged_when_rendering_is_unavailable(site, monkeypatch):
    monkeypatch.setattr(web, "_renderer", lambda url: None)
    doc = web.fetch_page("https://acme.example.com/pricing", "pricing", "Acme")
    assert doc is not None and doc.meta == {"thin": True}


def test_competitor_pricing_url_wins_over_discovered_link(site):
    acme = ACME.model_copy(update={"pricing_url": "https://acme.example.com/plans"})
    urls = {d.url for d in web.collect_official(acme) if d.source_type == "pricing"}
    assert urls == {"https://acme.example.com/plans"}


def test_fetch_page_returns_none_for_404_non_html_and_robots(site):
    assert web.fetch_page("https://acme.example.com/missing", "official", "Acme") is None
    assert web.fetch_page("https://acme.example.com/file.pdf", "docs", "Acme") is None
    assert web.fetch_page("https://acme.example.com/private/x", "official", "Acme") is None
    assert "/private/x" not in site["requested"]


def test_unreachable_homepage_gives_no_documents(site):
    gone = Competitor(name="Gone", website="https://acme.example.com/missing")
    assert web.collect_official(gone) == []


def test_find_links_ranks_same_site_pages_and_skips_noise():
    links = web.find_links(HOME, "https://acme.example.com/")
    assert links["pricing"] == ["https://acme.example.com/pricing"]
    assert links["features"] == [
        "https://acme.example.com/features",
        "https://acme.example.com/integrations",
    ]


def test_subdomains_count_as_the_same_site():
    html = '<a href="https://docs.acme.example.com/security">Security</a>'
    assert web.find_links(html, "https://www.acme.example.com/")["features"] == [
        "https://docs.acme.example.com/security"
    ]


def test_extract_text_keeps_title_and_pricing_table():
    html = page(
        "Plans",
        "<table><tr><th>Plan</th><th>Price</th></tr><tr><td>Pro</td><td>$12</td></tr></table>",
    )
    title, text = web.extract_text(html)
    assert title == "Plans"
    assert "$12" in text


def test_shallow_pages_first_and_at_most_two_per_section():
    html = (
        "".join(
            f'<a href="/integrations/{name}">{name} integration</a>'
            for name in ("asana", "slack", "hubspot")
        )
        + '<a href="/integrations">All integrations</a><a href="/features">Features</a>'
    )
    links = web.find_links(html, "https://acme.example.com/")["features"]
    assert links == [
        "https://acme.example.com/features",
        "https://acme.example.com/integrations",
        "https://acme.example.com/integrations/asana",
    ]


def test_pricing_pages_keep_plan_cards_that_main_content_extraction_drops(monkeypatch):
    cards = "<div>Pro $12 per seat</div><div>Team $20 per seat</div>"
    html = page("Pricing", f"{cards}<p>{FILLER}</p>")
    faq = f"Frequently asked questions. {FILLER}"  # what trafilatura sometimes keeps
    monkeypatch.setattr(web.trafilatura, "extract", lambda *a, **k: faq)
    assert "$12" not in web.extract_text(html)[1]
    assert "$12" in web.extract_text(html, keep_prices=True)[1]
    assert "$20" in web.extract_text(html, keep_prices=True)[1]
