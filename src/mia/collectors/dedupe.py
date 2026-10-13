"""Drop duplicate Documents (task B6): the same id (the same URL, or the same review of an app),
or the same text after normalization. The first copy wins, so order the input by preference.

Duplicates are per company. A comment that names two competitors is collected once for each, and
both copies stay: complaint extraction keeps only what each copy says about its own company."""

from __future__ import annotations

import hashlib
import re

from mia.schemas import Document

_WORD = re.compile(r"[a-z0-9]+")


def text_key(text: str) -> str:
    """Hash of the text ignoring case, punctuation and whitespace."""
    return hashlib.sha1(" ".join(_WORD.findall(text.lower())).encode()).hexdigest()


def dedupe(docs: list[Document]) -> list[Document]:
    seen_ids: set[tuple[str | None, str]] = set()
    seen_texts: set[tuple[str | None, str]] = set()
    unique = []
    for doc in docs:
        id_key, txt_key = (doc.company, doc.id), (doc.company, text_key(doc.text))
        if id_key in seen_ids or txt_key in seen_texts:
            continue
        seen_ids.add(id_key)
        seen_texts.add(txt_key)
        unique.append(doc)
    return unique
