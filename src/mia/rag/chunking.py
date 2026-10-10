"""Split Documents into Chunks (task C1).

- Short-form sources (reviews, comments, posts) stay whole: one document = one chunk.
- Long pages are split at Markdown headings. Small sections are packed together up to the
  target size, and oversized sections are cut into overlapping windows. Every window repeats its
  section heading, so it still makes sense on its own.
- Every chunk starts with a "[company | source | title]" header. Both BM25 and embeddings then
  see which company and source the text came from.

Sizes are in words: 300 words ≈ 400 tokens for English text.
Chunk ids are deterministic (f"{doc_id}:{i}"), so other modules can refer to them.
"""

from __future__ import annotations

from mia.schemas import COMPLAINT_SOURCE_TYPES, Chunk, Document

TARGET_WORDS = 300
OVERLAP_WORDS = 45  # 15%


def header(doc: Document) -> str:
    return f"[{doc.company or 'market'} | {doc.source_type} | {doc.title}]"


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]


def chunk_document(
    doc: Document, target_words: int = TARGET_WORDS, overlap_words: int = OVERLAP_WORDS
) -> list[Chunk]:
    text = doc.text.strip()
    if not text:
        return []
    if doc.source_type in COMPLAINT_SOURCE_TYPES and len(text.split()) <= 2 * target_words:
        pieces = [text]
    else:
        pieces = split_text(text, target_words, overlap_words)
    return [
        Chunk(
            id=f"{doc.id}:{i}",
            doc_id=doc.id,
            text=f"{header(doc)}\n{piece}",
            company=doc.company,
            source_type=doc.source_type,
            url=doc.url,
        )
        for i, piece in enumerate(pieces)
    ]


def split_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")
    units: list[str] = []
    for section in _sections(text):
        if len(section.split()) <= target_words:
            units.append(section)
        else:
            units.extend(_windows(section, target_words, overlap_words))
    return _pack(units, target_words)


def _sections(text: str) -> list[str]:
    """Split before every Markdown heading line; keep each heading with its body."""
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.lstrip().startswith("#") and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return [s for s in ("\n".join(lines).strip() for lines in sections) if s]


def _windows(section: str, target_words: int, overlap_words: int) -> list[str]:
    first_line, _, rest = section.partition("\n")
    heading = first_line.strip() if first_line.lstrip().startswith("#") else ""
    body = (rest if heading else section).split()
    room = target_words - len(heading.split())
    step = max(room - overlap_words, 1)
    windows = []
    for start in range(0, len(body), step):
        piece = " ".join(body[start : start + room])
        windows.append(f"{heading}\n{piece}" if heading else piece)
        if start + room >= len(body):
            break
    return windows


def _pack(units: list[str], target_words: int) -> list[str]:
    """Merge consecutive small units while the result stays within the target size."""
    packed: list[str] = []
    size = 0
    for unit in units:
        n = len(unit.split())
        if packed and size + n <= target_words:
            packed[-1] = f"{packed[-1]}\n\n{unit}"
            size += n
        else:
            packed.append(unit)
            size = n
    return packed
