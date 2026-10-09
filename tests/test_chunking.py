from __future__ import annotations

import pytest

from mia.rag.chunking import chunk_document, chunk_documents, header, split_text
from mia.schemas import Document
from tests.fixtures import load_documents


@pytest.fixture(scope="module")
def docs() -> dict[str, Document]:
    return {d.url: d for d in load_documents()}


def _body(chunk_text: str) -> str:
    return chunk_text.split("\n", 1)[1]


def test_short_form_doc_is_one_chunk_with_verbatim_text(docs):
    doc = docs["https://reddit.example.com/r/productivity/comments/n2"]
    [chunk] = chunk_document(doc)
    assert chunk.id == f"{doc.id}:0"
    assert chunk.text.startswith(f"{header(doc)}\n")
    assert _body(chunk.text) == doc.text
    assert (chunk.company, chunk.source_type, chunk.url) == (doc.company, "reddit", doc.url)


def test_long_page_splits_at_headings_within_target(docs):
    doc = docs["https://noted.example.com"]
    chunks = chunk_document(doc, target_words=120, overlap_words=18)
    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"{doc.id}:{i}" for i in range(len(chunks))]
    for c in chunks:
        assert len(_body(c.text).split()) <= 120
        assert _body(c.text).startswith("#")  # every chunk starts at a section heading


def test_no_words_are_lost(docs):
    doc = docs["https://noted.example.com"]
    chunked = set()
    for c in chunk_document(doc, target_words=80, overlap_words=12):
        chunked.update(_body(c.text).split())
    assert set(doc.text.split()) <= chunked


def test_oversized_section_gets_overlapping_windows_that_keep_the_heading():
    words = [f"w{i}" for i in range(50)]
    text = "## Section\n" + " ".join(words)
    pieces = split_text(text, target_words=21, overlap_words=5)
    assert all(p.startswith("## Section\n") for p in pieces)
    bodies = [p.split("\n", 1)[1].split() for p in pieces]
    assert bodies[0][-5:] == bodies[1][:5]  # 5-word overlap
    assert bodies[-1][-1] == "w49"


def test_small_sections_are_packed_together():
    text = "# A\none two\n\n# B\nthree four\n\n# C\nfive six"
    assert split_text(text, target_words=100, overlap_words=10) == [
        "# A\none two\n\n# B\nthree four\n\n# C\nfive six"
    ]


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        split_text("a b c", target_words=10, overlap_words=10)


def test_empty_document_has_no_chunks():
    doc = Document(id="x", url="u", source_type="official", company=None, text="   ")
    assert chunk_document(doc) == []


def test_all_fixture_chunk_ids_are_unique():
    chunks = chunk_documents(load_documents())
    assert len({c.id for c in chunks}) == len(chunks)
