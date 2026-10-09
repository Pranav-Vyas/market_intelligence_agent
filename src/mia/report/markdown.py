"""Render a MarketReport as Markdown. Section order mirrors Topic.md, plus sources and method."""

from __future__ import annotations

from collections import defaultdict

from mia.schemas import FeatureCell, MarketReport, Opportunity, PricingTier

STATUS_MARK = {"yes": "✓", "partial": "~", "no": "✗", "unknown": "?"}
BUCKETS = ["free", "pro", "business", "enterprise"]


class Citations:
    """Numbers source URLs in order of first use: [1], [2], ..."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def ref(self, url: str | None) -> str:
        if not url:
            return ""
        if url not in self._ids:
            self._ids[url] = len(self._ids) + 1
        return f"[{self._ids[url]}]"

    def items(self) -> list[tuple[int, str]]:
        return [(n, url) for url, n in self._ids.items()]


def _cell(cell: FeatureCell | None, cites: Citations) -> str:
    if cell is None:
        return STATUS_MARK["unknown"]
    ref = cites.ref(cell.url) if cell.status in ("yes", "partial") else ""
    return f"{STATUS_MARK[cell.status]}{ref}"


def _price(tier: PricingTier) -> str:
    if tier.monthly_usd is None:
        return "Contact sales" if tier.bucket == "enterprise" else tier.tier
    if tier.monthly_usd == 0:
        return "Free"
    text = f"${tier.monthly_usd:g}"
    if tier.per_seat:
        text += "/seat"
    return text + "/mo"


def _opportunity(
    n: int | None, o: Opportunity, report: MarketReport, cites: Citations, n_competitors: int
) -> list[str]:
    by_id = {c.id: c for c in report.complaints}
    heading = f"### {n}. {o.title}" if n is not None else f"### {o.title}"
    breakdown = " · ".join(f"{k} {v:.2f}" for k, v in o.score_breakdown.items())
    covered = round(o.coverage * n_competitors, 1)
    lines = [
        f"{heading} ({o.score:.1f} / 10)",
        "",
        f"- **Type:** {o.kind.replace('_', ' ')} · **Coverage:** {covered:g}/{n_competitors} "
        f"competitors · **Verified:** {'yes' if o.verified else 'no'}",
        f"- **Score breakdown:** {breakdown}",
        f"- **Why:** {o.rationale}",
    ]
    quotes = [by_id[e] for e in o.evidence if e in by_id]
    if quotes:
        lines.append("- **Evidence:**")
        for c in quotes:
            who = f"{c.company}, " if c.company else ""
            lines.append(f'  - "{c.quote}" ({who}{c.source_type}) {cites.ref(c.url)}')
    lines.append("")
    return lines


def render(report: MarketReport) -> str:
    cites = Citations()
    names = [c.name for c in report.competitors]
    n = len(names)
    out: list[str] = [
        f"# {report.market}: market report",
        "",
        f"Generated {report.generated_at:%Y-%m-%d %H:%M} UTC · {n} competitors · "
        f"{len(report.complaints)} complaints · {len(report.opportunities)} ranked opportunities",
        "",
    ]
    stubs = report.stats.get("stubs") or []
    if stubs:
        out += [
            f"> **Warning:** {len(stubs)} pipeline steps still return fake data: "
            + ", ".join(f"`{s}`" for s in stubs),
            "",
        ]

    # Competitors
    out += [
        "## Competitors",
        "",
        "| # | Company | Website | Mentions | What it does |",
        "|---|---|---|---|---|",
    ]
    for i, c in enumerate(report.competitors, 1):
        out.append(f"| {i} | {c.name} | {c.website} | {c.mention_count} | {c.one_liner} |")
    out.append("")

    # Feature comparison
    cells = {(c.company, c.feature): c for c in report.features}
    features = list(dict.fromkeys(c.feature for c in report.features))
    out += [
        "## Feature comparison",
        "",
        "| Feature | " + " | ".join(names) + " |",
        "|---" * (n + 1) + "|",
    ]
    for f in features:
        row = [_cell(cells.get((name, f)), cites) for name in names]
        out.append(f"| {f} | " + " | ".join(row) + " |")
    out += ["", "✓ offered · ~ partial · ✗ not offered · ? unknown (no evidence found)", ""]

    # Pricing comparison
    tiers: dict[tuple[str, str], list[PricingTier]] = defaultdict(list)
    for t in report.pricing:
        tiers[(t.company, t.bucket)].append(t)
    out += [
        "## Pricing comparison",
        "",
        "| Company | " + " | ".join(b.capitalize() for b in BUCKETS) + " |",
        "|---" * (len(BUCKETS) + 1) + "|",
    ]
    for name in names:
        url = next((t.source_url for t in report.pricing if t.company == name), None)
        row = ["; ".join(_price(t) for t in tiers[(name, b)]) or "—" for b in BUCKETS]
        out.append(f"| {name} {cites.ref(url)} | " + " | ".join(row) + " |")
    out.append("")

    # Customer complaints
    out += [
        "## Customer complaints",
        "",
        "| # | Pain point | Share | Severity (1–3) | Documents | Sources | Companies |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, cl in enumerate(sorted(report.clusters, key=lambda c: -c.share), 1):
        out.append(
            f"| {i} | {cl.label} | {cl.share:.0%} | {cl.mean_severity:.1f} | "
            f"{cl.distinct_docs} | {', '.join(cl.source_types)} | "
            f"{', '.join(cl.companies)} |"
        )
    out.append("")

    # Market gaps
    out += ["## Potential market gaps", ""]
    if not report.opportunities:
        out += ["No pain point passed the evidence gates. See weak signals below.", ""]
    for i, o in enumerate(report.opportunities, 1):
        out += _opportunity(i, o, report, cites, n)

    if report.weak_signals:
        out += [
            "## Weak signals",
            "",
            "These didn't pass the evidence gates (too few complaints, documents or source "
            "types), so they aren't ranked.",
            "",
        ]
        for o in report.weak_signals:
            out.append(f"- **{o.title}** ({o.score:.1f} / 10): {o.rationale}")
        out.append("")

    # Sources
    out += ["## Sources", ""]
    out += [f"[{num}] {url}  " for num, url in cites.items()]
    out.append("")

    # Method and stats
    llm = report.stats.get("llm", {})
    docs = report.stats.get("documents_by_source", {})
    out += [
        "## Method and run stats",
        "",
        "Gaps come only from clusters of real complaints. Each is scored on frequency (F), "
        "severity (S), competitor gap (G) and evidence strength (E), and must pass evidence gates "
        "before it is ranked. Every quote is checked against its source.",
        "",
        f"- Documents: {sum(docs.values())} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(docs.items()))
        + ")",
        f"- LLM: {llm.get('requests', 0)} requests, {llm.get('cache_hits', 0)} cache hits, "
        f"{llm.get('input_tokens', 0)} input / {llm.get('output_tokens', 0)} output tokens",
        f"- Runtime: {report.stats.get('runtime_s', 0):.1f} s",
        "",
    ]
    return "\n".join(out)
