"""Render a MarketReport as Markdown (task A11).

Section order mirrors Topic.md (competitors, features, pricing, complaints, market gaps), with
a short summary first and product failures, weak signals, sources and the method at the end.
Every claim cites its source as [n], numbered in order of first use and listed under "Sources".
"""

from __future__ import annotations

from collections import Counter, defaultdict

from mia import runtime
from mia.schemas import ComplaintCluster, FeatureCell, MarketReport, Opportunity, PricingTier

STATUS_MARK = {"yes": "✓", "partial": "~", "no": "✗", "unknown": "?"}
BUCKETS = ["free", "pro", "business", "enterprise"]
SOURCE_NAMES = {"app_store": "App Store", "hn": "Hacker News", "reddit": "Reddit"}
MAX_EVIDENCE_SHOWN = 5
MAX_WARNINGS_SHOWN = 10
QUOTE_CHARS = 160
SKEWED_SOURCE_SHARE = 0.6  # above this, the report says one source dominates
THIN_COMPANY_COMPLAINTS = 10  # below this, the report says a competitor's evidence is thin


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


def render(report: MarketReport) -> str:
    cites = Citations()
    out: list[str] = _header(report)
    out += _summary(report)
    out += _competitors(report)
    out += _features(report, cites)
    out += _pricing(report, cites)
    out += _complaints(report, cites)
    out += _gaps(report, cites)
    out += _failures(report, cites)
    out += _weak_signals(report)
    out += ["## Sources", ""] + [f"[{num}] {url}  " for num, url in cites.items()] + [""]
    out += _method(report)
    return "\n".join(out)


# ---- sections -----------------------------------------------------------------------------------


def _header(report: MarketReport) -> list[str]:
    out = [
        f"# {report.market}: market report",
        "",
        f"Generated {report.generated_at:%Y-%m-%d %H:%M} UTC · {len(report.competitors)} "
        f"competitors · {len(report.complaints)} complaints · "
        f"{_n(len(_market_gaps(report)), 'ranked market gap')} · "
        f"{_n(len(_product_failures(report)), 'product failure')}",
        "",
    ]
    stubs = report.stats.get("stubs") or []
    if stubs:
        out += [
            f"> **Warning:** {len(stubs)} pipeline steps still return fake data: "
            + ", ".join(f"`{s}`" for s in stubs),
            "",
        ]
    return out


def _summary(report: MarketReport) -> list[str]:
    out = ["## Summary", ""]
    names = ", ".join(c.name for c in report.competitors)
    out.append(
        f"- **Competitors:** {names}, found from comparison articles and checked against their "
        "own sites."
    )
    if report.clusters:
        top = sorted(report.clusters, key=lambda c: -c.share)[:3]
        out.append(
            "- **Most common complaints:** "
            + "; ".join(f"{c.label} ({c.share:.0%})" for c in top)
            + "."
        )
    gaps = _market_gaps(report)
    if gaps:
        out.append(
            "- **Top market gaps:** "
            + "; ".join(
                f"{o.title} ({o.score:.1f}/10, {o.kind.replace('_', ' ')})" for o in gaps[:3]
            )
            + "."
        )
    else:
        out.append("- **Market gaps:** no pain point passed the evidence gates in this run.")
    failures = _product_failures(report)
    if failures:
        out.append(
            "- **Most reported product failures:** "
            + "; ".join(o.title for o in failures[:3])
            + ". These are things that break, not openings for a new product."
        )
    return out + _evidence_limits(report) + [""]


def _evidence_limits(report: MarketReport) -> list[str]:
    """What the data can't support, stated next to the conclusions drawn from it."""
    out = []
    by_source = Counter(c.source_type for c in report.complaints)
    if by_source:
        source, count = by_source.most_common(1)[0]
        if count / len(report.complaints) > SKEWED_SOURCE_SHARE:
            out.append(
                f"- **Source skew:** {count / len(report.complaints):.0%} of complaints come from "
                f"{SOURCE_NAMES.get(source, source)}, so its users' problems are over-represented. "
                "Frequency scores average over sources to offset this."
            )
    by_company = Counter(c.company for c in report.complaints)
    thin = [
        f"{c.name} ({by_company[c.name]})"
        for c in report.competitors
        if by_company[c.name] < THIN_COMPANY_COMPLAINTS
    ]
    if thin:
        out.append(
            f"- **Thin evidence:** few public complaints were found for {', '.join(thin)}, so the "
            "gaps say little about them."
        )
    return out


def _competitors(report: MarketReport) -> list[str]:
    out = [
        "## Competitors",
        "",
        "| # | Company | Website | Mentions | What it does |",
        "|---|---|---|---|---|",
    ]
    for i, c in enumerate(report.competitors, 1):
        out.append(f"| {i} | {c.name} | {c.website} | {c.mention_count} | {c.one_liner} |")
    return out + [""]


def _features(report: MarketReport, cites: Citations) -> list[str]:
    names = [c.name for c in report.competitors]
    cells = {(c.company, c.feature): c for c in report.features}
    features = list(dict.fromkeys(c.feature for c in report.features))
    out = [
        "## Feature comparison",
        "",
        "From each company's own website.",
        "",
        "| Feature | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for f in features:
        row = [_cell(cells.get((name, f)), cites) for name in names]
        out.append(f"| {f} | " + " | ".join(row) + " |")
    return out + ["", "✓ offered · ~ partial · ✗ not offered · ? unknown (no evidence found)", ""]


def _pricing(report: MarketReport, cites: Citations) -> list[str]:
    tiers: dict[tuple[str, str], list[PricingTier]] = defaultdict(list)
    for t in report.pricing:
        tiers[(t.company, t.bucket)].append(t)
    out = [
        "## Pricing comparison",
        "",
        "US dollars per month as shown on each pricing page; the per-month price when billed "
        "annually is in brackets. Prices that couldn't be found on the page are left out.",
        "",
        "| Company | " + " | ".join(b.capitalize() for b in BUCKETS) + " |",
        "|---" * (len(BUCKETS) + 1) + "|",
    ]
    for c in report.competitors:
        url = next((t.source_url for t in report.pricing if t.company == c.name), None)
        row = ["; ".join(_price(t) for t in tiers[(c.name, b)]) or "—" for b in BUCKETS]
        out.append(f"| {c.name} {cites.ref(url)} | " + " | ".join(row) + " |")
    return out + [""]


def _complaints(report: MarketReport, cites: Citations) -> list[str]:
    by_id = {c.id: c for c in report.complaints}
    out = [
        "## Customer complaints",
        "",
        f"{len(report.complaints)} complaints, each quoted word for word from a public post or "
        "review, grouped by the problem they describe.",
        "",
        "| # | Pain point | Share | Severity (1–3) | Docs | Sources | Companies | Example |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, cl in enumerate(sorted(report.clusters, key=lambda c: -c.share), 1):
        example = next((by_id[cid] for cid in cl.complaint_ids if cid in by_id), None)
        quote = f'"{_short(example.quote)}" {cites.ref(example.url)}' if example else ""
        out.append(
            f"| {i} | {cl.label} | {cl.share:.0%} | {cl.mean_severity:.1f} | "
            f"{cl.distinct_docs} | {_sources(cl.source_types)} | {', '.join(cl.companies)} | "
            f"{quote} |"
        )
    return out + [""]


def _gaps(report: MarketReport, cites: Citations) -> list[str]:
    out = ["## Potential market gaps", ""]
    gaps = _market_gaps(report)
    if not gaps:
        return out + ["No pain point passed the evidence gates. See weak signals below.", ""]
    out += [
        "Things customers want that a product could offer, ranked by score (0–10): frequency F, "
        "severity S, competitor gap G and evidence strength E. An *unmet need* is one few "
        "competitors claim to address; *poor execution* means vendors claim it, yet customers "
        "still complain.",
        "",
    ]
    clusters = {c.id: c for c in report.clusters}
    for i, o in enumerate(gaps, 1):
        out += _opportunity(i, o, clusters.get(o.cluster_id), report, cites)
    return out


def _failures(report: MarketReport, cites: Citations) -> list[str]:
    failures = _product_failures(report)
    if not failures:
        return []
    out = [
        "## Product failures",
        "",
        "Things that break for customers across several products: crashes, lost data, billing "
        "and support problems. They are ranked apart from market gaps because they show where "
        "today's products fall down, not what's missing from the market. Here G is the share of "
        "competitors whose customers report the problem.",
        "",
    ]
    clusters = {c.id: c for c in report.clusters}
    for i, o in enumerate(failures, 1):
        out += _opportunity(i, o, clusters.get(o.cluster_id), report, cites)
    return out


def _market_gaps(report: MarketReport) -> list[Opportunity]:
    return [o for o in report.opportunities if o.kind != "product_failure"]


def _product_failures(report: MarketReport) -> list[Opportunity]:
    return [o for o in report.opportunities if o.kind == "product_failure"]


def _opportunity(
    n: int, o: Opportunity, cl: ComplaintCluster | None, report: MarketReport, cites: Citations
) -> list[str]:
    by_id = {c.id: c for c in report.complaints}
    n_comp = len(report.competitors)
    breakdown = " · ".join(f"{k} {v:.2f}" for k, v in o.score_breakdown.items())
    lines = [f"### {n}. {o.title} ({o.score:.1f} / 10)", ""]
    if o.kind != "product_failure":
        lines.append(f"- **Kind:** {o.kind.replace('_', ' ')}")
    if cl is not None:
        counts = f"{_n(len(cl.complaint_ids), 'complaint')}, {_n(cl.distinct_docs, 'document')}"
        lines.append(
            f"- **Frequency:** {cl.share:.0%} of complaints ({counts}, {_sources(cl.source_types)})"
        )
        lines.append(f"- **Severity:** {cl.mean_severity:.1f} / 3")
        lines.append(f"- **Reported for:** {', '.join(cl.companies)}")
    if o.kind == "product_failure":
        lines.append(f"- **Spread:** {o.rationale}")
    else:
        lines.append(
            f"- **Coverage:** {o.coverage * n_comp:g}/{n_comp} competitors claim to address it. "
            f"{o.rationale}"
        )
    lines.append(f"- **Score:** {o.score:.1f} / 10 ({breakdown})")
    quotes = [by_id[e] for e in o.evidence if e in by_id][:MAX_EVIDENCE_SHOWN]
    if quotes:
        mark = " ✓ verified" if o.verified else ""
        lines.append(f"- **Evidence**{mark}:")
        for c in quotes:
            who = f"{c.company}, " if c.company else ""
            src = SOURCE_NAMES.get(c.source_type, c.source_type)
            lines.append(f'  - "{c.quote}" ({who}{src}) {cites.ref(c.url)}')
    elif not o.verified:
        lines.append("- **Evidence:** no quote passed verification")
    return lines + [""]


def _weak_signals(report: MarketReport) -> list[str]:
    if not report.weak_signals:
        return []
    clusters = {c.id: c for c in report.clusters}
    out = [
        "## Weak signals",
        "",
        "Real complaints that didn't pass an evidence gate, so they aren't ranked.",
        "",
    ]
    for o in report.weak_signals:
        cl = clusters.get(o.cluster_id)
        why = _failed_gates(cl) if cl else ""
        tag = ", product failure" if o.kind == "product_failure" else ""
        out.append(f"- **{o.title}** ({o.score:.1f} / 10{tag}){': ' + why if why else ''}")
    return out + [""]


def _method(report: MarketReport) -> list[str]:
    s = runtime.settings()
    stats = report.stats
    llm = stats.get("llm", {})
    docs = stats.get("documents_by_source", {})
    v = stats.get("verification") or {}
    out = [
        "## Method and run stats",
        "",
        '1. **Competitors** come from comparison and "best of" articles, ranked by how many '
        "independent sources mention them, and kept only if their homepage is a product in this "
        "market.",
        "2. **Features and pricing** come from each company's own pages. Every feature cell and "
        "price must be found in the page text, or it is left out.",
        "3. **Complaints** are extracted from Hacker News comments and low-rated App Store "
        "reviews. Each one quotes the customer word for word; quotes that can't be found in the "
        "source are dropped.",
        "4. **Pain points** are clusters of similar complaints, each marked a product failure or "
        "a need. For needs, coverage asks whether each competitor's own site claims to address "
        "the problem.",
        f"5. **Gaps** are scored in code, never by the LLM. A pain point is ranked only with at "
        f"least {s.gate_min_complaints} complaints from {s.gate_min_docs} documents about "
        f"{s.gate_min_companies} companies, and at most {s.max_complaints_per_doc} complaints per "
        f"document count. Market gaps also need {s.gate_min_source_types} source types.",
        "6. **Verification:** every quote shown under a gap is checked word for word against its "
        "source and judged by an LLM for whether it supports the gap.",
        "",
        f"- Documents: {sum(docs.values())} ("
        + ", ".join(f"{k} {n}" for k, n in sorted(docs.items()))
        + ")",
        "- Complaints: "
        + ", ".join(
            f"{SOURCE_NAMES.get(k, k)} {n}"
            for k, n in Counter(c.source_type for c in report.complaints).most_common()
        ),
    ]
    if v:
        out.append(
            f"- Verification: {v.get('opportunities_verified', 0)}/{v.get('opportunities', 0)} "
            f"ranked pain points verified; {v.get('quotes_passed', 0)}/"
            f"{v.get('quotes_checked', 0)} "
            "quotes passed"
            + (f" ({v['quote_pass_rate']:.0%})" if v.get("quote_pass_rate") is not None else "")
        )
    agent = stats.get("agent")
    if agent:
        out.append(
            f"- Agents: {agent.get('discovery_rounds', 0)} discovery rounds, "
            f"{max(agent.get('verify_rounds', 1) - 1, 0)} rounds of follow-up research"
        )
        researched = agent.get("researched", [])
        out += [f"  - {line.removeprefix('after research: ')}" for line in researched]
    out += [
        f"- LLM: {llm.get('requests', 0)} requests, {llm.get('cache_hits', 0)} cache hits, "
        f"{llm.get('input_tokens', 0)} input / {llm.get('output_tokens', 0)} output tokens",
        f"- Runtime: {stats.get('runtime_s', 0):.1f} s",
    ]
    if llm.get("fallbacks"):
        out.append(f"- {llm['fallbacks']} calls fell back from the smart to the bulk model")
    warnings = stats.get("warnings") or []
    if warnings:
        out.append(f"- {len(warnings)} recovered problems:")
        out += [f"  - {w}" for w in warnings[:MAX_WARNINGS_SHOWN]]
        if len(warnings) > MAX_WARNINGS_SHOWN:
            out.append(f"  - ... and {len(warnings) - MAX_WARNINGS_SHOWN} more (see run.json)")
    return out + [""]


# ---- helpers ------------------------------------------------------------------------------------


def _cell(cell: FeatureCell | None, cites: Citations) -> str:
    if cell is None:
        return STATUS_MARK["unknown"]
    ref = cites.ref(cell.url) if cell.status in ("yes", "partial") else ""
    return f"{STATUS_MARK[cell.status]}{ref}"


def _price(tier: PricingTier) -> str:
    if tier.monthly_usd is None and tier.annual_monthly_usd is None:
        return "Contact sales" if tier.bucket == "enterprise" else tier.tier
    if tier.monthly_usd == 0 or (tier.monthly_usd is None and tier.annual_monthly_usd == 0):
        return "Free"
    seat = "/seat" if tier.per_seat else ""
    monthly = f"${tier.monthly_usd:g}{seat}/mo" if tier.monthly_usd is not None else ""
    annual = f"${tier.annual_monthly_usd:g}" if tier.annual_monthly_usd else ""
    if monthly and annual and tier.annual_monthly_usd != tier.monthly_usd:
        return f"{monthly} ({annual})"
    return monthly or f"{annual}{seat}/mo (annual)"


def _n(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _sources(source_types: list[str]) -> str:
    return " + ".join(SOURCE_NAMES.get(s, s) for s in source_types)


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUOTE_CHARS else text[: QUOTE_CHARS - 1].rstrip() + "…"


def _failed_gates(cl: ComplaintCluster) -> str:
    s = runtime.settings()
    reasons = []
    if len(cl.complaint_ids) < s.gate_min_complaints:
        reasons.append(f"{_n(len(cl.complaint_ids), 'complaint')} (needs {s.gate_min_complaints})")
    if cl.distinct_docs < s.gate_min_docs:
        reasons.append(f"{_n(cl.distinct_docs, 'document')} (needs {s.gate_min_docs})")
    if len(cl.companies) < s.gate_min_companies:
        reasons.append(
            f"only {', '.join(cl.companies) or 'no company'} (needs {s.gate_min_companies} "
            "companies)"
        )
    if cl.category != "failure" and len(cl.source_types) < s.gate_min_source_types:
        reasons.append(
            f"only {_sources(cl.source_types)} (needs {s.gate_min_source_types} source types)"
        )
    return "; ".join(reasons)
