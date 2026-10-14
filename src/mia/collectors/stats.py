"""Collection report (task B8): documents per company per source, and what falls short of the
minimums a competitor needs before its analysis means anything (SPEC.md §5)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from mia.config import Settings
from mia.schemas import COMPLAINT_SOURCE_TYPES, Chunk


@dataclass
class CompanyStats:
    company: str
    docs_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def complaint_docs(self) -> int:
        return sum(n for s, n in self.docs_by_source.items() if s in COMPLAINT_SOURCE_TYPES)

    def gaps(self, s: Settings) -> list[str]:
        out = []
        if not self.docs_by_source.get("official"):
            out.append("no official pages")
        if not self.docs_by_source.get("pricing"):
            out.append("no pricing page")
        if self.complaint_docs < s.min_complaint_docs_per_company:
            out.append(
                f"{self.complaint_docs} complaint docs (minimum {s.min_complaint_docs_per_company})"
            )
        return out


def collection_stats(chunks: list[Chunk]) -> list[CompanyStats]:
    """Distinct documents per company and source, in the order companies first appear."""
    docs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for ch in chunks:
        docs[ch.company or "(market)"][ch.source_type].add(ch.doc_id)
    return [
        CompanyStats(company, {source: len(ids) for source, ids in by_source.items()})
        for company, by_source in docs.items()
    ]
