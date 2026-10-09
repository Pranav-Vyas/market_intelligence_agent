"""Official pages: homepage, features, integrations and pricing, cleaned to text (task B4)."""

from __future__ import annotations

from mia import fake
from mia.schemas import Competitor, Document, SourceType, doc_id
from mia.stubs import stub


@stub
def fetch_page(url: str, source_type: SourceType, company: str | None) -> Document | None:
    text = fake.official_text(company or "This market")
    return Document(id=doc_id(url), url=url, source_type=source_type, company=company, text=text)


@stub
def collect_official(c: Competitor) -> list[Document]:
    pricing_url = c.pricing_url or f"{c.website.rstrip('/')}/pricing"
    return [
        Document(
            id=doc_id(c.website),
            url=c.website,
            source_type="official",
            company=c.name,
            title=f"{c.name}: home",
            text=fake.official_text(c.name),
        ),
        Document(
            id=doc_id(pricing_url),
            url=pricing_url,
            source_type="pricing",
            company=c.name,
            title=f"{c.name}: pricing",
            text=fake.pricing_text(c.name),
        ),
    ]
