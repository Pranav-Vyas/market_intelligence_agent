"""Drop duplicate Documents (task B6): the same id (the same URL, or the same review of an app),
or the same text after normalization. The first copy wins, so order the input by preference."""

from __future__ import annotations

import hashlib
import re

from mia.schemas import Document

_WORD = re.compile(r"[a-z0-9]+")


def text_key(text: str) -> str:
    """Hash of the text ignoring case, punctuation and whitespace."""
    return hashlib.sha1(" ".join(_WORD.findall(text.lower())).encode()).hexdigest()


def dedupe(docs: list[Document]) -> list[Document]:
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    unique = []
    for doc in docs:
        key = text_key(doc.text)
        if doc.id in seen_ids or key in seen_texts:
            continue
        seen_ids.add(doc.id)
        seen_texts.add(key)
        unique.append(doc)
    return unique
