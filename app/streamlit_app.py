"""Market Gap Scout browser (tasks C11, C12, C14, C16).

Pick a market and a run, then read it in tabs: Competitors, Features, Pricing, Complaints, Gaps,
Evidence, Agent trace and Report. The trace can be browsed per scout and per competitor, next to
the graph diagram. Evidence searches the market's saved hybrid index live.

Run with:
    uv sync --extra ui
    uv run streamlit run app/streamlit_app.py

Everything shown comes from scraped pages and reviews, so all of it is HTML-escaped before it
goes into the page's custom HTML.
"""

from __future__ import annotations

import contextlib
import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from mia.config import get_settings

st.set_page_config(page_title="Market Gap Scout", layout="wide", initial_sidebar_state="expanded")

# ── look ──────────────────────────────────────────────────────────────────────────────────────────
# A reading tool for checking evidence: paper, slate ink, one blue for links and scores. Chart
# colors come from a palette validated for colour-blind readers; the blue above is for text only.
PAPER = "#F6F7F9"
INK = "#1D2733"
MUTED = "#5B6573"
RULE = "#D9DEE4"
LINK = "#1F5A8A"
NEED = "#2A78D6"  # chart: market gaps and anything neutral
FAILURE = "#EB6834"  # chart: product failures
VERIFIED = "#2E7D4F"
SERIES = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]
STATUS_COLOR = {"yes": "#2E7D4F", "partial": "#A36A00", "no": "#B3261E", "unknown": "#8A939E"}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea,
.stApp [data-baseweb="select"] div, .stApp [data-testid="stMetricValue"] div {{
    font-family: 'Public Sans', system-ui, sans-serif;
}}
table.grid {{ font-variant-numeric: tabular-nums; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.2rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
[data-testid="stSidebar"] {{ background: #ECEFF3; border-right: 1px solid {RULE}; }}
h1 {{ font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
h2, h3, h4 {{ font-weight: 600; color: {INK}; }}
a, a:visited {{ color: {LINK}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; }}
[data-testid="stMetricValue"] div {{ color: {INK}; font-weight: 600; font-size: 1.75rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.lede {{ color: {MUTED}; margin: -0.6rem 0 1.4rem; }}
.finding {{
    background: #FFFFFF;
    border: 1px solid {RULE};
    border-radius: 6px;
    padding: 1.1rem 1.3rem 0.9rem;
    margin-bottom: 1rem;
}}
.finding-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
.finding-title {{ font-size: 1.08rem; font-weight: 600; color: {INK}; }}
.finding-score {{ font-size: 1.4rem; font-weight: 700; color: {LINK}; white-space: nowrap; }}
.finding-score small {{ font-size: 0.8rem; font-weight: 500; color: {MUTED}; }}
.facts {{ color: {MUTED}; font-size: 0.88rem; margin: 0.25rem 0 0.7rem; }}
.tag {{
    display: inline-block; border: 1px solid {RULE}; border-radius: 3px;
    padding: 0 0.4rem; font-size: 0.78rem; color: {MUTED}; margin-right: 0.35rem;
}}
.tag-verified {{ color: {VERIFIED}; border-color: {VERIFIED}; }}
.tag-failure {{ color: #9A4A12; border-color: #D9A27F; }}
.anatomy {{ display: flex; height: 8px; gap: 2px; margin: 0.2rem 0 0.25rem; max-width: 420px; }}
.anatomy span {{ display: block; border-radius: 1px; }}
.anatomy-key {{ font-size: 0.78rem; color: {MUTED}; margin-bottom: 0.6rem; }}
.anatomy-key i {{
    display: inline-block; width: 9px; height: 9px; border-radius: 1px;
    margin: 0 0.25rem 0 0.7rem; vertical-align: baseline;
}}
.anatomy-key i:first-child {{ margin-left: 0; }}
.rationale {{ font-size: 0.92rem; color: {INK}; margin: 0.4rem 0 0.6rem; }}
.quote-block {{
    border-left: 2px solid {RULE};
    padding: 0.2rem 0 0.2rem 0.85rem;
    margin: 0.5rem 0;
    color: {INK};
    font-size: 0.92rem;
    line-height: 1.5;
}}
.quote-block small {{ color: {MUTED}; font-size: 0.8rem; }}
.weak {{ font-size: 0.92rem; margin: 0.3rem 0; }}
.weak span {{ color: {MUTED}; }}
table.grid {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; background: #FFFFFF; }}
table.grid th {{
    text-align: center; font-weight: 600; color: {MUTED}; padding: 0.45rem 0.7rem;
    border-bottom: 1px solid {RULE};
}}
table.grid th:first-child, table.grid td:first-child {{ text-align: left; }}
table.grid td {{ text-align: center; padding: 0.4rem 0.7rem; border-bottom: 1px solid #EEF1F4; }}
table.grid td.empty {{ color: #A5ADB7; }}
table.grid a {{ text-decoration: none; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── constants & helpers ───────────────────────────────────────────────────────────────────────────
SETTINGS = get_settings()
RUNS_DIR = SETTINGS.runs_dir  # follows MIA_DATA_DIR, like the CLI
SOURCE_LABEL = {
    "hn": "Hacker News",
    "app_store": "App Store",
    "reddit": "Reddit",
    "official": "Official",
    "pricing": "Pricing",
    "docs": "Docs",
    "review_site": "Review site",
    "comparison": "Comparison",
    "news": "News",
}
STATUS_ICON = {"yes": "✓", "partial": "~", "no": "✗", "unknown": "?"}
STATUS_WORD = {"yes": "offered", "partial": "partly", "no": "not offered", "unknown": "unknown"}
BUCKET_ORDER = ["free", "pro", "business", "enterprise", "other"]
COMPONENTS = {"F": "frequency", "S": "severity", "G": "gap", "E": "evidence"}
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"


def find_markets() -> dict[str, list[Path]]:
    """Market slug → its run.json files, newest first; the market with the newest run first."""
    if not RUNS_DIR.exists():
        return {}
    newest = sorted(RUNS_DIR.rglob("run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    markets: dict[str, list[Path]] = {}
    for path in newest:
        markets.setdefault(path.parent.parent.name, []).append(path)
    return markets


def run_label(path: Path) -> str:
    """ "20261105-184252" → "2026-11-05 18:42"."""
    stamp = path.parent.name
    if len(stamp) == 15 and stamp[8] == "-":
        return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
    return stamp


@st.cache_data(show_spinner=False)
def load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace(run_json_path: Path) -> list[dict]:
    tp = run_json_path.parent / "trace.jsonl"
    if not tp.exists():
        return []
    rows = []
    for line in tp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    return rows


def short(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def esc(text: object) -> str:
    """Scraped text is data: escape it before it goes into the page's HTML."""
    return html.escape(str(text), quote=True)


def source_link(url: str | None) -> str:
    return f' · <a href="{esc(url)}" target="_blank">source</a>' if url else ""


def cite(company: str | None, source_type: str) -> str:
    """ "Otter · App Store", the line under a quote."""
    who = f"{company} · " if company else ""
    return esc(who) + esc(SOURCE_LABEL.get(source_type, source_type))


def failed_gates(cl: dict) -> str:
    s = SETTINGS
    out = []
    if len(cl.get("complaint_ids", [])) < s.gate_min_complaints:
        out.append(f"{len(cl['complaint_ids'])} of {s.gate_min_complaints} complaints")
    if cl.get("distinct_docs", 0) < s.gate_min_docs:
        out.append(f"{cl['distinct_docs']} of {s.gate_min_docs} documents")
    if len(cl.get("companies", [])) < s.gate_min_companies:
        out.append(f"{len(cl['companies'])} of {s.gate_min_companies} companies")
    needs_sources = cl.get("category", "need") != "failure"
    if needs_sources and len(cl.get("source_types", [])) < s.gate_min_source_types:
        out.append(f"{len(cl['source_types'])} of {s.gate_min_source_types} source types")
    return ", ".join(out)


def index_version(market_slug: str) -> float | None:
    """When the market's index was saved, so a newer run's index replaces the cached one."""
    saved = SETTINGS.indexes_dir / market_slug / "chunks.json"
    return saved.stat().st_mtime if saved.exists() else None


@st.cache_resource(show_spinner="Loading the evidence index…", max_entries=3)
def load_evidence_index(market_slug: str, version: float):
    from mia.pipeline import load_index

    return load_index(market_slug)[1]


@st.cache_data(show_spinner="Searching…", max_entries=100)
def search_evidence(
    market_slug: str,
    version: float,
    query: str,
    company: str | None,
    sources: tuple[str, ...],
    k: int,
    rerank: bool,
) -> list[dict]:
    hits = load_evidence_index(market_slug, version).search(
        query, k=k, company=company, source_types=list(sources) or None, rerank=rerank
    )
    return [
        {"score": hit.score, "retriever": hit.retriever, **hit.chunk.model_dump()} for hit in hits
    ]


def fmt_price(tier: dict) -> str:
    m, a = tier.get("monthly_usd"), tier.get("annual_monthly_usd")
    seat = "/seat" if tier.get("per_seat") else ""
    if m == 0 or a == 0:
        return "Free"
    if m is None and a is None:
        return "Contact sales" if tier.get("bucket") == "enterprise" else tier.get("tier", "—")
    parts = []
    if m is not None:
        parts.append(f"${m:g}{seat}/mo")
    if a is not None and a != m:
        parts.append(f"(${a:g} annual)")
    return " ".join(parts) or "—"


def style_chart(fig, height: int, **layout) -> None:
    """One quiet look for every chart: no background, hairline grid, the page's type."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        font=dict(family="Public Sans, system-ui, sans-serif", color=INK, size=13),
        margin=dict(t=10, b=40, l=10, r=10),
        height=height,
        **layout,
    )
    fig.update_xaxes(gridcolor="#E6E9ED", zerolinecolor=RULE)
    fig.update_yaxes(gridcolor="#E6E9ED", zerolinecolor=RULE)


def bar_labels(titles: list[str], width: int = 34) -> list[str]:
    """Long pain point names wrapped onto two lines instead of cut off mid-word."""
    out = []
    for title in titles:
        words, lines = title.split(), [""]
        for w in words:
            if len(lines[-1]) + len(w) + 1 > width and lines[-1]:
                lines.append(w)
            else:
                lines[-1] = f"{lines[-1]} {w}".strip()
        out.append("<br>".join(lines[:2]) + ("…" if len(lines) > 2 else ""))
    return out


# ── sidebar ───────────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Market Gap Scout")

    markets = find_markets()
    if not markets:
        st.warning('No runs yet. Run `uv run mia run "AI meeting assistants"` first.')
        st.stop()

    market_names = {
        slug: load_report(str(paths[0])).get("market", slug) for slug, paths in markets.items()
    }
    market_slug = st.selectbox("Market", list(markets), format_func=market_names.get, index=0)
    run_paths = markets[market_slug]
    selected_path = st.selectbox("Run", run_paths, format_func=run_label, index=0)
    report = load_report(str(selected_path))

    stats = report.get("stats", {})
    llm_info = stats.get("llm", {})
    engine = "Multi-agent graph" if stats.get("engine") == "graph" else "Linear pipeline"
    lines = [
        f"{engine}",
        f"{llm_info.get('requests', 0)} new LLM requests, "
        f"{llm_info.get('cache_hits', 0)} from cache",
    ]
    if stats.get("runtime_s") is not None:
        lines.append(f"Took {stats['runtime_s']:.0f} seconds")
    st.caption("  \n".join(lines))
    run_warnings = stats.get("warnings") or []
    if run_warnings:
        st.warning(
            f"{len(run_warnings)} recovered problems in this run, so it may be missing evidence. "
            "The Report tab lists them."
        )
    stubs_used = stats.get("stubs") or []
    if stubs_used:
        st.warning(f"{len(stubs_used)} steps in this run still return fake data.")

# ── unpack report ─────────────────────────────────────────────────────────────────────────────────
market: str = report.get("market", "Unknown market")
competitors: list[dict] = report.get("competitors", [])
features: list[dict] = report.get("features", [])
pricing: list[dict] = report.get("pricing", [])
complaints: list[dict] = report.get("complaints", [])
clusters: list[dict] = report.get("clusters", [])
opportunities: list[dict] = report.get("opportunities", [])
market_gaps = [o for o in opportunities if o.get("kind") != "product_failure"]
product_failures = [o for o in opportunities if o.get("kind") == "product_failure"]
weak_signals: list[dict] = report.get("weak_signals", [])

complaint_by_id = {c["id"]: c for c in complaints}
cluster_by_id = {cl["id"]: cl for cl in clusters}


def score_anatomy(opp: dict) -> str:
    """The score as the four parts it adds up from, each as wide as its share of 10 points."""
    weights = SETTINGS.score_weights
    parts = opp.get("score_breakdown", {})
    spans = []
    for key, color in zip(COMPONENTS, SERIES, strict=False):
        points = 10 * weights.get(key, 0) * parts.get(key, 0)
        if points > 0:
            spans.append(
                f'<span style="width:{points * 10:.1f}%;background:{color}" '
                f'title="{COMPONENTS[key]}: {points:.1f} of the score"></span>'
            )
    return f'<div class="anatomy">{"".join(spans)}</div>'


def anatomy_key() -> None:
    swatches = "".join(
        f'<i style="background:{color}"></i>{name}'
        for name, color in zip(COMPONENTS.values(), SERIES, strict=False)
    )
    st.markdown(
        f'<div class="anatomy-key">Score out of 10, split into {swatches}</div>',
        unsafe_allow_html=True,
    )


def opportunity_cards(items: list[dict]) -> None:
    n_comp = len(competitors) or 1
    for i, opp in enumerate(items, 1):
        cl = cluster_by_id.get(opp["cluster_id"], {})
        kind = opp.get("kind", "")
        if kind == "product_failure":
            tags = '<span class="tag tag-failure">product failure</span>'
            reach = f"reported for {len(cl.get('companies', []))} of {n_comp} competitors"
        else:
            name = "unmet need" if kind == "unmet_need" else "poor execution"
            tags = f'<span class="tag">{name}</span>'
            claims = round(opp["coverage"] * n_comp * 2) / 2  # coverage is stored rounded
            reach = f"{claims:g} of {n_comp} competitors claim to solve it"
        tags += (
            '<span class="tag tag-verified">verified</span>'
            if opp.get("verified")
            else '<span class="tag">not verified</span>'
        )
        quotes = "".join(
            f'<div class="quote-block">“{esc(short(ev["quote"], 220))}”<br>'
            f"<small>{cite(ev.get('company'), ev.get('source_type', ''))}"
            f"{source_link(ev.get('url'))}</small></div>"
            for ev in (complaint_by_id.get(e) for e in opp.get("evidence", [])[:4])
            if ev
        )
        facts = (
            f"{cl.get('share', 0):.0%} of complaints, {cl.get('distinct_docs', 0)} documents, "
            f"severity {cl.get('mean_severity', 0):.1f} of 3, {reach}"
        )
        st.markdown(
            f"""<div class="finding">
<div class="finding-head"><div class="finding-title">{i}. {esc(opp["title"])}</div>
<div class="finding-score">{opp["score"]:.1f}<small> / 10</small></div></div>
{score_anatomy(opp)}
<div class="facts">{tags} {esc(facts)}</div>
<div class="rationale">{esc(opp.get("rationale", ""))}</div>
{quotes}
</div>""",
            unsafe_allow_html=True,
        )


# ── header ────────────────────────────────────────────────────────────────────────────────────────
st.markdown(f"# {market}")
generated = report.get("generated_at", "")
with contextlib.suppress(ValueError):
    generated = datetime.fromisoformat(generated).strftime("%-d %B %Y")
st.markdown(
    f'<p class="lede">Report from {esc(generated)}, built from {len(competitors)} competitors\' '
    f"own pages and {len(complaints)} customer complaints.</p>",
    unsafe_allow_html=True,
)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Competitors", len(competitors))
c2.metric("Complaints", len(complaints))
c3.metric("Pain points", len(clusters))
c4.metric("Ranked gaps", len(market_gaps))
c5.metric("Product failures", len(product_failures))
vr = stats.get("verification_rate")
c6.metric("Verified", f"{vr:.0%}" if vr is not None else "—")

# ── tabs ──────────────────────────────────────────────────────────────────────────────────────────
(
    tab_gaps,
    tab_comp,
    tab_feat,
    tab_price,
    tab_compl,
    tab_evidence,
    tab_trace,
    tab_report,
) = st.tabs(
    [
        "Gaps",
        "Competitors",
        "Features",
        "Pricing",
        "Complaints",
        "Evidence",
        "Agent trace",
        "Report",
    ]
)

# ── gaps ──────────────────────────────────────────────────────────────────────────────────────────
with tab_gaps:
    st.markdown("### Ranked market gaps")
    if not market_gaps:
        st.info("No market gap passed the evidence gates in this run. See the weak signals below.")
    if opportunities:
        ver = stats.get("verification") or {}
        if ver:
            st.caption(
                f"{ver.get('opportunities_verified', 0)} of {ver.get('opportunities', 0)} ranked "
                f"pain points are verified: {ver.get('quotes_passed', 0)} of "
                f"{ver.get('quotes_checked', 0)} quotes were found word for word in their source "
                "and judged to support the pain point."
            )
        anatomy_key()
        opportunity_cards(market_gaps)
        if product_failures:
            st.markdown("### Product failures")
            st.caption(
                "Things that break for customers across several products. They show where "
                "today's products fall down, not what's missing from the market, so they are "
                "ranked apart from the gaps."
            )
            opportunity_cards(product_failures)

    if weak_signals:
        st.markdown("### Weak signals")
        st.caption("Real complaints that didn't pass an evidence gate, so they aren't ranked.")
        st.markdown(
            "".join(
                f'<div class="weak">{esc(ws["title"])} '
                f"<span>{ws['score']:.1f}, "
                f"{esc(failed_gates(cluster_by_id.get(ws['cluster_id'], {})) or 'unranked')}"
                f"</span></div>"
                for ws in weak_signals
            ),
            unsafe_allow_html=True,
        )

# ── competitors ───────────────────────────────────────────────────────────────────────────────────
with tab_comp:
    st.markdown("### Competitors")
    if not competitors:
        st.info("This run found no competitors.")
    else:
        st.caption(
            "Found in comparison and best-of articles, then checked against their own sites. "
            "Mentions counts how many separate articles named each one."
        )
        rows = "".join(
            f"<tr><td><a href='{esc(c['website'])}' target='_blank'>{esc(c['name'])}</a></td>"
            f"<td>{c.get('mention_count', 0)}</td>"
            f"<td style='text-align:left'>{esc(c.get('one_liner', ''))}</td></tr>"
            for c in competitors
        )
        st.markdown(
            "<table class='grid'><thead><tr><th>Company</th><th>Mentions</th>"
            f"<th style='text-align:left'>What it does</th></tr></thead><tbody>{rows}</tbody>"
            "</table>",
            unsafe_allow_html=True,
        )

# ── features ──────────────────────────────────────────────────────────────────────────────────────
with tab_feat:
    st.markdown("### Feature comparison")
    st.caption(
        "From each company's own pages: ✓ offered, ~ partly, ✗ not offered, ? not mentioned. "
        "Hover a mark for the quote it rests on, or click it to open the page."
    )
    if not features:
        st.info("This run has no feature data.")
    else:
        feat_names = list(dict.fromkeys(f["feature"] for f in features))
        comp_names = [c["name"] for c in competitors] or list(
            dict.fromkeys(f["company"] for f in features)
        )
        cell_map = {(f["company"], f["feature"]): f for f in features}
        header = "<th>Feature</th>" + "".join(f"<th>{esc(n)}</th>" for n in comp_names)
        body = ""
        for feat in feat_names:
            cells = ""
            for comp in comp_names:
                cell = cell_map.get((comp, feat))
                status = cell.get("status", "unknown") if cell else "unknown"
                mark = STATUS_ICON.get(status, "?")
                tip = esc(short(cell.get("quote") or "", 160)) if cell else ""
                label = f"{esc(comp)}: {STATUS_WORD.get(status, status)}. {tip}".strip()
                styled = f'<span style="color:{STATUS_COLOR.get(status, MUTED)}">{mark}</span>'
                if cell and cell.get("url") and status in ("yes", "partial", "no"):
                    styled = f'<a href="{esc(cell["url"])}" target="_blank">{styled}</a>'
                cells += f'<td title="{label}">{styled}</td>'
            body += f"<tr><td>{esc(feat)}</td>{cells}</tr>"
        st.markdown(
            f"<div style='overflow-x:auto'><table class='grid'><thead><tr>{header}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>",
            unsafe_allow_html=True,
        )

# ── pricing ───────────────────────────────────────────────────────────────────────────────────────
with tab_price:
    st.markdown("### Pricing comparison")
    st.caption(
        "US dollars per month as each pricing page shows them, with the per-month price when "
        "billed annually in brackets. Prices that couldn't be found on the page are left out."
    )
    if not pricing:
        st.info("This run has no pricing data.")
    else:
        price_map: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for tier in pricing:
            price_map[tier["company"]][tier["bucket"]].append(tier)
        comp_names_p = [c["name"] for c in competitors] or list(price_map.keys())
        header_p = "<th>Company</th>" + "".join(f"<th>{b.capitalize()}</th>" for b in BUCKET_ORDER)
        rows_p = ""
        for comp in comp_names_p:
            cells_p = ""
            for bucket in BUCKET_ORDER:
                tiers = price_map[comp].get(bucket, [])
                if tiers:
                    cells_p += f"<td>{'<br>'.join(esc(fmt_price(t)) for t in tiers)}</td>"
                else:
                    cells_p += "<td class='empty'>—</td>"
            page = next((t["source_url"] for t in pricing if t["company"] == comp), None)
            name = f"<a href='{esc(page)}' target='_blank'>{esc(comp)}</a>" if page else esc(comp)
            rows_p += f"<tr><td>{name}</td>{cells_p}</tr>"
        st.markdown(
            f"<div style='overflow-x:auto'><table class='grid'><thead><tr>{header_p}</tr></thead>"
            f"<tbody>{rows_p}</tbody></table></div>",
            unsafe_allow_html=True,
        )

# ── complaints ────────────────────────────────────────────────────────────────────────────────────
with tab_compl:
    st.markdown("### Customer complaints")
    if not clusters:
        st.info("This run has no complaint groups.")
    else:
        st.caption(
            f"{len(complaints)} complaints, each quoted word for word, grouped into pain points. "
            "Blue are needs a product could meet, orange are product failures."
        )
        try:
            import plotly.graph_objects as go

            ordered = sorted(clusters, key=lambda c: c["share"])[-12:]
            fig = go.Figure(
                go.Bar(
                    x=[c["share"] * 100 for c in ordered],
                    y=bar_labels([c["label"].capitalize() for c in ordered], 40),
                    orientation="h",
                    marker=dict(
                        color=[FAILURE if c.get("category") == "failure" else NEED for c in ordered]
                    ),
                    width=0.55,
                    hovertemplate="%{x:.1f}% of complaints<extra></extra>",
                )
            )
            style_chart(fig, 40 * len(ordered) + 60, xaxis_title="Share of complaints (%)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        except ImportError:
            pass

        for cl in sorted(clusters, key=lambda c: c["share"], reverse=True):
            kind = "product failure" if cl.get("category") == "failure" else "need"
            with st.expander(
                f"{cl['label'].capitalize()}: {cl['share']:.0%} of complaints, "
                f"severity {cl['mean_severity']:.1f}, {cl['distinct_docs']} documents"
            ):
                sources = ", ".join(SOURCE_LABEL.get(s, s) for s in cl["source_types"])
                st.caption(
                    f"A {kind}. From {sources}, about {', '.join(cl.get('companies') or ['—'])}."
                )
                shown = [complaint_by_id[c] for c in cl["complaint_ids"] if c in complaint_by_id]
                st.markdown(
                    "".join(
                        f'<div class="quote-block">“{esc(short(c["quote"], 200))}”<br>'
                        f"<small>{cite(c.get('company'), c.get('source_type', ''))}"
                        f"{source_link(c.get('url'))}</small></div>"
                        for c in shown[:3]
                    ),
                    unsafe_allow_html=True,
                )

# ── evidence ──────────────────────────────────────────────────────────────────────────────────────
with tab_evidence:
    st.markdown("### Evidence explorer")
    st.caption(
        "Searches everything collected for this market: keyword and meaning matches, then "
        "reranked. It reads the market's latest index, which may be newer than an older run."
    )
    slug = market_slug
    version = index_version(slug)
    if version is None:
        st.info(
            f"No saved index for {slug}. Run `uv run mia index` for this market, or restore a "
            "snapshot with `uv run mia snapshot restore`."
        )
    else:
        chunks = load_evidence_index(slug, version).chunks
        companies = sorted({ch.company for ch in chunks if ch.company})
        source_types = sorted({ch.source_type for ch in chunks})
        with st.form("evidence_search"):
            query = st.text_input(
                "Search the evidence", placeholder="too expensive for small teams"
            )
            f1, f2, f3, f4 = st.columns([2, 3, 2, 1])
            company = f1.selectbox("Company", ["All", *companies], key="evidence_company")
            sources = f2.multiselect(
                "Sources",
                source_types,
                format_func=lambda s: SOURCE_LABEL.get(s, s),
                key="evidence_sources",
            )
            k = f3.slider("Results", 5, 30, 10, key="evidence_k")
            rerank = f4.checkbox("Rerank", value=True, key="evidence_rerank")
            st.form_submit_button("Search")
        if query.strip():
            hits = search_evidence(
                slug,
                version,
                query.strip(),
                None if company == "All" else company,
                tuple(sources),
                k,
                rerank,
            )
            st.caption(f"{len(hits)} passages from {len(chunks)} in the index")
            if not hits:
                st.info("Nothing matches this search and these filters.")
            for hit in hits:
                body = hit["text"].split("\n", 1)[-1]  # drop the "[company | source]" header
                body = re.sub(r"\*\*|^#+\s*|\s#+\s", " ", body, flags=re.M)  # page markdown
                st.markdown(
                    f'<div class="quote-block">{esc(short(body, 600))}<br>'
                    f"<small>{cite(hit.get('company'), hit['source_type'])}"
                    f" · {esc(hit['retriever'])} {hit['score']:.2f}"
                    f"{source_link(hit.get('url'))}</small></div>",
                    unsafe_allow_html=True,
                )

# ── agent trace ───────────────────────────────────────────────────────────────────────────────────
with tab_trace:
    st.markdown("### Agent trace")
    trace = load_trace(selected_path)
    if not trace:
        st.info(
            "This run has no trace: it was made with the linear pipeline (`mia run --linear`). "
            "Runs of the multi-agent graph (`mia run`) write one."
        )
    else:
        agent_stats = stats.get("agent") or {}
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Graph steps", max((r.get("step") or 0) for r in trace))
        a2.metric("Discovery rounds", agent_stats.get("discovery_rounds", "—"))
        a3.metric("Research rounds", max(agent_stats.get("verify_rounds", 1) - 1, 0))
        a4.metric("New LLM requests", sum(r.get("requests", 0) for r in trace))
        researched = agent_stats.get("researched") or []
        if researched:
            st.markdown("**What follow-up research changed**")
            st.markdown(
                "".join(
                    f'<div class="quote-block">{esc(line.removeprefix("after research: "))}</div>'
                    for line in researched
                ),
                unsafe_allow_html=True,
            )

        diagram = selected_path.parent / "graph.mmd"
        if diagram.exists():
            with st.expander("Graph diagram"):
                # the diagram is written by our own graph code (graph.mmd) and escaped
                st.iframe(
                    f'<div class="mermaid">{esc(diagram.read_text(encoding="utf-8"))}</div>'
                    f'<script src="{MERMAID_JS}"></script>'
                    "<script>mermaid.initialize({startOnLoad: true, theme: 'neutral'});</script>",
                    height=520,
                )

        agents = sorted({r.get("agent", "") for r in trace if r.get("agent")})
        names = sorted({r["competitor"] for r in trace if r.get("competitor")})
        f1, f2 = st.columns(2)
        sel_agent = f1.selectbox("Scout / agent", ["All"] + agents)
        sel_comp = f2.selectbox("Competitor", ["All"] + names)
        filtered = [
            r
            for r in trace
            if (sel_agent == "All" or r.get("agent") == sel_agent)
            and (sel_comp == "All" or r.get("competitor") == sel_comp)
        ]
        if sel_comp != "All":
            st.markdown(f"#### {esc(sel_comp)}: what each scout did")
        by_step: dict[Any, list[dict]] = defaultdict(list)
        for r in filtered:
            by_step[r.get("step")].append(r)
        for step in sorted(by_step, key=lambda x: (x is None, x)):
            rows = by_step[step]
            parallel = len({(r.get("agent"), r.get("competitor")) for r in rows}) > 1
            st.markdown(
                f"**Step {step if step is not None else '?'}**"
                + (f", {len(rows)} actions in parallel" if parallel else "")
            )
            st.dataframe(
                [
                    {
                        "agent": r.get("agent", ""),
                        "competitor": r.get("competitor") or "—",
                        "action": r.get("action", ""),
                        "result": r.get("result_summary", ""),
                        "requests": r.get("requests", 0),
                        "tokens": r.get("tokens", 0),
                    }
                    for r in rows
                ],
                width="stretch",
                hide_index=True,
            )
        st.caption(f"{len(trace)} trace entries in total")

# ── report ────────────────────────────────────────────────────────────────────────────────────────
with tab_report:
    report_md = selected_path.parent / "report.md"
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8")
        st.download_button("Download report.md", text, file_name="report.md")
        st.markdown(text)  # rendered without unsafe HTML
    else:
        st.info("There's no report.md next to this run.")
