"""Competitor discovery: listicle/"alternatives" search → LLM extraction → mention ranking →
homepage validation → top K (task A5)."""

from __future__ import annotations

from mia import fake
from mia.schemas import Competitor
from mia.stubs import stub


@stub
def discover_competitors(market: str, k: int = 8) -> list[Competitor]:
    return fake.competitors(k)
