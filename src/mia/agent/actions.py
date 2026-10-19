"""The fixed actions a scout may choose (SPEC.md §3). The LLM picks one per step with
`llm().extract(prompt, schema=...)`; code validates it before running it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NextAction(BaseModel):
    """A Customer Scout's next step."""

    action: Literal["search", "done"]
    source: str = ""  # one of COMPLAINT_SOURCES, for "search"
    query: str = ""  # 1-4 words, for "search"
    reason: str = ""  # one sentence: why this action


class PagePick(BaseModel):
    link_id: int
    kind: Literal["pricing", "features", "integrations", "security", "other"]


class PagePicks(BaseModel):
    """A Product Scout's choice of site links to fetch."""

    pages: list[PagePick] = Field(default_factory=list)


class NextQueries(BaseModel):
    """A Discovery Scout's new search queries for one more round."""

    queries: list[str] = Field(default_factory=list)
    reason: str = ""
