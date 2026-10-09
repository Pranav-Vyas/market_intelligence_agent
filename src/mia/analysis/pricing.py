"""Pricing extraction from pricing pages. Every price must appear in the page text (task A8)."""

from __future__ import annotations

from mia import fake
from mia.schemas import Competitor, Document, PricingTier
from mia.stubs import stub


@stub
def extract_pricing(c: Competitor, docs: list[Document]) -> list[PricingTier]:
    url = docs[0].url if docs else (c.pricing_url or c.website)
    pro = 8 + fake.pick(c.name, 15)
    return [
        PricingTier(
            company=c.name,
            tier="Free",
            bucket="free",
            monthly_usd=0.0,
            limits=["300 minutes per month"],
            source_url=url,
        ),
        PricingTier(
            company=c.name,
            tier="Pro",
            bucket="pro",
            monthly_usd=float(pro),
            per_seat=True,
            source_url=url,
        ),
        PricingTier(
            company=c.name,
            tier="Business",
            bucket="business",
            monthly_usd=float(pro * 2),
            per_seat=True,
            source_url=url,
        ),
        PricingTier(company=c.name, tier="Enterprise", bucket="enterprise", source_url=url),
    ]
