"""The research graph's state (task A12, SPEC.md §7).

Plain data only (Pydantic models, lists, dicts), so the SQLite checkpointer can save it after
every step. The fields parallel branches write to use an `operator.add` reducer: each branch
appends its own items and none overwrites another's. The search index is not part of the state;
the Analyst builds it from `documents` and saves it to disk for the Verifier.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from mia.schemas import (
    Competitor,
    Complaint,
    ComplaintCluster,
    Document,
    FeatureCell,
    Opportunity,
    PricingTier,
)


class ResearchState(TypedDict, total=False):
    run_id: str
    market: str
    offline: bool
    started_at: float  # epoch seconds, so the runtime survives a resume
    competitors: list[Competitor]
    discovery: dict  # the Discovery Scout's progress between rounds (plain data)
    discovery_round: int
    documents: Annotated[list[Document], operator.add]  # appended by parallel scouts
    pricing: Annotated[list[PricingTier], operator.add]
    complaints: Annotated[list[Complaint], operator.add]  # the Analyst appends only new ids
    extracted_docs: list[str]  # documents whose complaints were already extracted
    features: list[FeatureCell]
    clusters: list[ComplaintCluster]
    opportunities: list[Opportunity]
    weak_signals: list[Opportunity]
    verification: dict
    verify_round: int
    research: list[dict]  # the Verifier's targets for the next research round
    run_dir: str
    trace: Annotated[list[dict], operator.add]


class ScoutTask(TypedDict):
    """What `Send` passes to one Product or Customer Scout branch."""

    run_id: str
    market: str
    competitor: Competitor
    position: int  # the competitor's rank, used to put branch results back in a fixed order
    focus: str | None  # targeted research: the pain point to dig into
    known: list[str]  # ids of this competitor's documents from earlier rounds (with a focus)


SCHEMA_TYPES = (
    Competitor,
    Complaint,
    ComplaintCluster,
    Document,
    FeatureCell,
    Opportunity,
    PricingTier,
)  # the only classes the checkpointer may rebuild when it loads a saved state
