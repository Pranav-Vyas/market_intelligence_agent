"""Fake but internally consistent data for the stubs, so the pipeline runs end to end on Day 1.

Delete this module once no stub imports it any more.
"""

from __future__ import annotations

import hashlib

from mia.schemas import Competitor, Document, doc_id

COMPETITORS = [
    ("Example Notes", "notes"),
    ("Sample Scribe", "scribe"),
    ("Demo Minutes", "minutes"),
    ("Mock Recap", "recap"),
    ("Placeholder AI", "placeholder"),
]

FEATURES = [
    "Transcription",
    "Speaker identification",
    "Meeting summaries",
    "Action items",
    "CRM integration",
    "Calendar integration",
]

# (normalized summary, verbatim sentence a customer might write, severity)
COMPLAINT_THEMES = [
    (
        "too expensive for small teams",
        "Honestly the Pro plan is way too expensive for a three person team.",
        2,
    ),
    (
        "duplicate or missing calendar events",
        "It keeps creating duplicate events in Google Calendar after every meeting.",
        2,
    ),
    (
        "inaccurate transcription with accents",
        "Transcription falls apart as soon as someone with an accent speaks.",
        3,
    ),
    (
        "bot joining calls raises privacy concerns",
        "Clients get uncomfortable when the bot joins the call without asking.",
        3,
    ),
]


def pick(key: str, n: int) -> int:
    """Deterministic pseudo-random index in [0, n) derived from `key`."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


def competitors(k: int) -> list[Competitor]:
    return [
        Competitor(
            name=name,
            website=f"https://{slug}.example.com",
            pricing_url=f"https://{slug}.example.com/pricing",
            one_liner=f"{name} records, transcribes and summarizes meetings.",
            mention_count=len(COMPETITORS) - i,
        )
        for i, (name, slug) in enumerate(COMPETITORS[:k])
    ]


def official_text(company: str) -> str:
    offered = [f for f in FEATURES if pick(company + f, 3) != 0]
    return f"{company} gives you " + ", ".join(offered) + " for every meeting."


def pricing_text(company: str) -> str:
    pro = 8 + pick(company, 15)
    return (
        f"{company} pricing. Free: 300 minutes per month. Pro: ${pro} per seat per month. "
        f"Business: ${pro * 2} per seat per month. Enterprise: contact sales."
    )


def complaint_doc(company: str, source: str, i: int) -> Document:
    themes = [
        COMPLAINT_THEMES[pick(f"{company}{source}{i}{j}", len(COMPLAINT_THEMES))] for j in range(2)
    ]
    text = " ".join(dict.fromkeys(sentence for _, sentence, _ in themes))
    url = f"https://{source}.example.com/{company.lower().replace(' ', '-')}/{i}"
    return Document(
        id=doc_id(url),
        url=url,
        source_type=source,  # type: ignore[arg-type]
        company=company,
        title=f"{company} feedback #{i}",
        text=text,
    )
