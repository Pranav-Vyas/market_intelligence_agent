"""Complaint extraction: batched LLM calls over complaint chunks, verbatim quotes only,
severity per the rubric in SPEC.md §4 (tasks A6, A10)."""

from __future__ import annotations

from mia import fake
from mia.schemas import COMPLAINT_SOURCE_TYPES, Chunk, Complaint
from mia.stubs import stub


@stub
def extract_complaints(chunks: list[Chunk]) -> list[Complaint]:
    complaints = []
    for ch in chunks:
        if ch.source_type not in COMPLAINT_SOURCE_TYPES:
            continue
        for j, (summary, sentence, severity) in enumerate(fake.COMPLAINT_THEMES):
            if sentence in ch.text:
                complaints.append(
                    Complaint(
                        id=f"{ch.id}:{j}",
                        company=ch.company,
                        summary=summary,
                        quote=sentence,
                        severity=severity,
                        chunk_id=ch.id,
                        doc_id=ch.doc_id,
                        source_type=ch.source_type,
                        url=ch.url,
                    )
                )
    return complaints
