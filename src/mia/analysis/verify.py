"""Verification (task C10): (1) every quote must appear verbatim in its chunk, (2) a batched LLM
judge checks that each quote supports the opportunity. Sets `verified` and drops failing
evidence."""

from __future__ import annotations

from mia.rag.hybrid import HybridIndex
from mia.schemas import Complaint, Opportunity
from mia.stubs import stub


@stub
def verify(
    opps: list[Opportunity], complaints: list[Complaint], index: HybridIndex
) -> list[Opportunity]:
    known = {c.id for c in complaints}
    out = []
    for o in opps:
        evidence = [e for e in o.evidence if e in known]
        out.append(o.model_copy(update={"evidence": evidence, "verified": len(evidence) >= 3}))
    return out
