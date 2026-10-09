"""Shared data contracts. Every module talks to the others through these models.

Owner: A. Change this file only in a small dedicated PR, and tell the team (SPEC.md §11.6).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal[
    "official", "pricing", "docs", "reddit", "hn", "app_store", "review_site", "comparison", "news"
]

# Source types that carry customer complaints (as opposed to vendor claims).
COMPLAINT_SOURCE_TYPES: frozenset[str] = frozenset({"reddit", "hn", "app_store", "review_site"})


def utcnow() -> datetime:
    return datetime.now(UTC)


def doc_id(url: str, suffix: str = "") -> str:
    """Stable Document id: sha1 of the URL, plus a suffix for sub-items such as comments."""
    return hashlib.sha1(f"{url}#{suffix}".encode()).hexdigest()[:16]


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str


class Competitor(BaseModel):
    name: str
    website: str
    pricing_url: str | None = None
    one_liner: str = ""
    mention_count: int = 0  # how many discovery sources mentioned it


class Document(BaseModel):
    id: str  # sha1(url [+ suffix for comments])
    url: str
    source_type: SourceType
    company: str | None  # None = market-level page (e.g. a listicle)
    title: str = ""
    text: str
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    meta: dict = Field(default_factory=dict)  # rating, upvotes, author_hash, ...


class Chunk(BaseModel):
    id: str  # f"{doc_id}:{i}"
    doc_id: str
    text: str  # starts with a header: "[Otter | reddit | <title>]"
    company: str | None
    source_type: SourceType
    url: str


class Evidence(BaseModel):
    chunk: Chunk
    score: float
    retriever: str  # "bm25" | "dense" | "hybrid" | "rerank"


class FeatureCell(BaseModel):
    company: str
    feature: str
    status: Literal["yes", "partial", "no", "unknown"]  # unknown ≠ no
    quote: str | None = None
    chunk_id: str | None = None
    url: str | None = None


class PricingTier(BaseModel):
    company: str
    tier: str  # as named by the vendor
    bucket: Literal["free", "pro", "business", "enterprise", "other"]
    monthly_usd: float | None = None  # per seat if per_seat
    annual_monthly_usd: float | None = None
    per_seat: bool | None = None
    limits: list[str] = Field(default_factory=list)
    source_url: str


class Complaint(BaseModel):
    id: str
    company: str | None
    summary: str  # normalized: "inaccurate transcription with accents"
    quote: str  # VERBATIM span from the chunk
    severity: int = Field(ge=1, le=3)  # rubric in SPEC.md §4
    chunk_id: str
    doc_id: str
    source_type: SourceType
    url: str


class ComplaintCluster(BaseModel):
    id: str
    label: str
    complaint_ids: list[str]
    share: float  # fraction of all complaints
    distinct_docs: int
    source_types: list[SourceType]
    mean_severity: float
    companies: list[str]


class Opportunity(BaseModel):
    title: str
    cluster_id: str
    kind: Literal["unmet_need", "poor_execution"]
    coverage: float  # 0..1, share of competitors that address it well
    score: float  # 0..10
    score_breakdown: dict[str, float]
    rationale: str
    evidence: list[str]  # complaint ids whose quotes support this opportunity
    verified: bool = False


class MarketReport(BaseModel):
    market: str
    generated_at: datetime = Field(default_factory=utcnow)
    competitors: list[Competitor]
    features: list[FeatureCell]
    pricing: list[PricingTier]
    complaints: list[Complaint]
    clusters: list[ComplaintCluster]
    opportunities: list[Opportunity]
    weak_signals: list[Opportunity]
    stats: dict  # docs per source, LLM requests/tokens, verification rate, runtime, stubs used
