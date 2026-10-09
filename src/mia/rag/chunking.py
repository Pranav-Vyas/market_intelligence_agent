"""Split Documents into Chunks (task C1).

Real rules (SPEC.md §9, C1): web pages ~400 tokens with 15% overlap, split on headings;
one review or comment = one chunk; every chunk starts with a "[company | source | title]" header.
Chunk ids must be deterministic (f"{doc_id}:{i}") so other modules can refer to them.
"""

from __future__ import annotations

from mia.schemas import Chunk, Document
from mia.stubs import stub


def header(doc: Document) -> str:
    return f"[{doc.company or 'market'} | {doc.source_type} | {doc.title}]"


@stub
def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [
        Chunk(
            id=f"{d.id}:0",
            doc_id=d.id,
            text=f"{header(d)} {d.text}",
            company=d.company,
            source_type=d.source_type,
            url=d.url,
        )
        for d in docs
    ]
