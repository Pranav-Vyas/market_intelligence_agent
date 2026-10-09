from __future__ import annotations

import pytest
from pydantic import ValidationError

from mia.schemas import Complaint, Document, doc_id


def test_doc_id_is_stable_and_suffix_sensitive():
    assert doc_id("https://a.com") == doc_id("https://a.com")
    assert doc_id("https://a.com") != doc_id("https://a.com", "comment-1")
    assert len(doc_id("https://a.com")) == 16


def test_document_round_trips_through_json():
    d = Document(id="x", url="https://a.com", source_type="reddit", company="Otter", text="hi")
    assert Document.model_validate_json(d.model_dump_json()) == d


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValidationError):
        Document(id="x", url="u", source_type="twitter", company=None, text="t")


def test_complaint_severity_must_be_1_to_3():
    with pytest.raises(ValidationError):
        Complaint(
            id="c",
            company=None,
            summary="s",
            quote="q",
            severity=4,
            chunk_id="k",
            doc_id="d",
            source_type="hn",
            url="u",
        )
