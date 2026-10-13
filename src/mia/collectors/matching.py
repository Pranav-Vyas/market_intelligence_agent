"""Helpers shared by the complaint collectors: is this text about the product, and author hashing.

Product names are often ordinary words or jokes (Fellow, Granola, "can't fathom", "TL;DV"). A
text counts as being about a competitor when it contains the product's domain ("otter.ai"), or
when it names the product close to a word from the market name ("meeting"). A plain-word name
must be capitalized in the middle of a sentence, which is how people write product names and
not how they write "my fellow engineers".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from urllib.parse import urlsplit

from mia.discovery import name_key
from mia.schemas import Competitor

MIN_PLAIN_NAME_CHARS = 5  # "read" (from Read.ai) is too common to match on its own
MIN_MARKET_TERM_CHARS = 4  # drops "ai"
NEAR_CHARS = 200  # how close a market word must be to the name, about two sentences

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = ".!?:;\n"


def author_hash(username: str) -> str:
    """Usernames are never stored (SPEC.md §5); this keeps one author's posts linkable."""
    return hashlib.sha256(username.encode()).hexdigest()[:12]


def site(url: str) -> str:
    """Host without "www.": "https://www.otter.ai/pricing" → "otter.ai"."""
    host = urlsplit(url if "//" in url else f"//{url}").netloc.lower()
    return host.removeprefix("www.")


def market_terms(market: str) -> list[str]:
    """Significant words of the market name: "AI meeting assistants" → meeting, assistant."""
    terms = []
    for word in _WORD.findall(market.lower()):
        word = word[:-1] if word.endswith("s") and not word.endswith("ss") else word
        if len(word) >= MIN_MARKET_TERM_CHARS and word not in terms:
            terms.append(word)
    return terms


def plain_name(c: Competitor) -> str | None:
    """The name as people write it in passing ("otter"), or None when that is too ambiguous."""
    key = name_key(c.name)
    is_single_word = c.name.strip().isalnum()
    return key if key and (is_single_word or len(key) >= MIN_PLAIN_NAME_CHARS) else None


def written_names(c: Competitor) -> list[str]:
    """Spellings of a name that has punctuation or spaces: "tl;dv" → tl;dv, "tl dv", tldv."""
    name = c.name.lower().strip()
    if name.isalnum():
        return []
    spellings = [name, " ".join(_WORD.findall(name)), "".join(_WORD.findall(name))]
    return [s for s in dict.fromkeys(spellings) if s]


def is_about(c: Competitor, market: str, text: str, title: str = "") -> bool:
    low_title, low_text = title.lower(), text.lower()
    domain = site(c.website)
    if has_word(low_title, domain) or has_word(low_text, domain):
        return True
    terms = market_terms(market)
    plain = plain_name(c)
    names = written_names(c) + ([plain] if plain else [])
    if any(has_word(low_title, n) for n in names) and any(t in low_title for t in terms):
        return True
    mentions = [m for n in written_names(c) for m in _find(low_text, n)]
    if plain:
        mentions += [m for m in _find(text, plain.capitalize()) if _mid_sentence(text, m)]
    return any(
        term in low_text[max(0, m.start() - NEAR_CHARS) : m.end() + NEAR_CHARS]
        for m in mentions
        for term in terms
    )


def has_word(text: str, word: str) -> bool:
    """Whole-word match; `text` must already be lowercase."""
    return bool(word) and next(_find(text, word), None) is not None


def _find(text: str, word: str) -> Iterator[re.Match]:
    return re.finditer(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text)


def _mid_sentence(text: str, match: re.Match) -> bool:
    before = text[: match.start()].rstrip(" ")
    return bool(before) and before[-1] not in _SENTENCE_END
