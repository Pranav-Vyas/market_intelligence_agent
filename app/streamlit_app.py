"""Market Gap Scout browser (tasks C11, C12).

Pick a run → tabs for Competitors / Features / Pricing / Complaints / Gaps / Agent trace /
Report. The trace can be browsed per scout and per competitor, next to the graph diagram.

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from mia.config import get_settings

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Gap Scout",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p { color: #8b949e; font-size: 0.82rem; }
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #58a6ff; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8b949e; }
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-weight: 500;
    padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #ffffff !important;
}
.opp-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}
.opp-card:hover { border-color: #388bfd; }
.opp-title { font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: .4rem; }
.opp-score {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #fff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .82rem;
    font-weight: 600;
    margin-bottom: .6rem;
}
.badge {
    display: inline-block; border-radius: 20px; padding: 2px 10px;
    font-size: .75rem; font-weight: 600; margin-right: 4px;
}
.badge-unmet  { background: #1f3a52; color: #58a6ff; }
.badge-poor   { background: #3a1f1f; color: #ff7b72; }
.badge-verified   { background: #1a3a2a; color: #3fb950; }
.badge-unverified { background: #2e2e1f; color: #d29922; }
.quote-block {
    background: #0d1117;
    border-left: 3px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: .6rem 1rem;
    margin: .4rem 0;
    color: #c9d1d9;
    font-size: .88rem;
    font-style: italic;
}
.stat-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: .75rem;
    color: #8b949e;
    margin-right: 6px;
}
.cell-yes     { color: #3fb950; font-weight: 700; }
.cell-partial { color: #d29922; font-weight: 600; }
.cell-no      { color: #f85149; font-weight: 600; }
.cell-unknown { color: #6e7681; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── constants & helpers ────────────────────────────────────────────────────────
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
STATUS_CLASS = {
    "yes": "cell-yes",
    "partial": "cell-partial",
    "no": "cell-no",
    "unknown": "cell-unknown",
}
BUCKET_ORDER = ["free", "pro", "business", "enterprise", "other"]
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"


def find_runs() -> list[tuple[str, Path]]:
    results = []
    if not RUNS_DIR.exists():
        return results
    for p in sorted(RUNS_DIR.rglob("run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        market_slug = p.parent.parent.name
        timestamp = p.parent.name
        results.append((f"{market_slug}  ·  {timestamp}", p))
    return results


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


def failed_gates(cl: dict) -> str:
    s = SETTINGS
    out = []
    if len(cl.get("complaint_ids", [])) < s.gate_min_complaints:
        out.append(f"{len(cl['complaint_ids'])}/{s.gate_min_complaints} complaints")
    if cl.get("distinct_docs", 0) < s.gate_min_docs:
        out.append(f"{cl['distinct_docs']}/{s.gate_min_docs} documents")
    if len(cl.get("source_types", [])) < s.gate_min_source_types:
        out.append(f"{len(cl['source_types'])}/{s.gate_min_source_types} source types")
    return ", ".join(out)


def badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{text}</span>'


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


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 Market Gap Scout")
    st.markdown("---")

    runs = find_runs()
    if not runs:
        st.warning(
            'No completed runs found.\n\nRun `uv run mia run "AI meeting assistants"` first.',
            icon="⚠️",
        )
        st.stop()

    labels = [label for label, _ in runs]
    choice = st.selectbox("**Select a run**", labels, index=0)
    selected_path = next(p for lbl, p in runs if lbl == choice)
    report = load_report(str(selected_path))

    st.markdown("---")
    st.markdown("**Run info**")
    gen = report.get("generated_at", "")[:16].replace("T", " ")
    st.caption(f"🕐 {gen} UTC")
    stats = report.get("stats", {})
    llm_info = stats.get("llm", {})
    reqs = llm_info.get("requests", 0)
    hits = llm_info.get("cache_hits", 0)
    st.caption(f"⚡ {reqs} LLM calls  ·  {hits} cached")
    rt = stats.get("runtime_s")
    if rt is not None:
        st.caption(f"⏱ {rt:.1f} s runtime")
    st.caption(f"🔧 Engine: {stats.get('engine', 'linear')}")
    stubs_used = stats.get("stubs") or []
    if stubs_used:
        st.warning(f"⚠️ {len(stubs_used)} stub(s) still active", icon="🧪")
    st.markdown("---")
    st.caption("Market Gap Scout")

# ── unpack report ──────────────────────────────────────────────────────────────
market: str = report.get("market", "Unknown market")
competitors: list[dict] = report.get("competitors", [])
features: list[dict] = report.get("features", [])
pricing: list[dict] = report.get("pricing", [])
complaints: list[dict] = report.get("complaints", [])
clusters: list[dict] = report.get("clusters", [])
opportunities: list[dict] = report.get("opportunities", [])
weak_signals: list[dict] = report.get("weak_signals", [])

complaint_by_id = {c["id"]: c for c in complaints}
cluster_by_id = {cl["id"]: cl for cl in clusters}

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.markdown(f"# 🔭 {market}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Competitors", len(competitors))
c2.metric("Complaints", len(complaints))
c3.metric("Clusters", len(clusters))
c4.metric("Ranked Gaps", len(opportunities))
vr = stats.get("verification_rate")
c5.metric("Verification rate", f"{vr:.0%}" if vr is not None else "—")
st.markdown("---")

# ── tabs ───────────────────────────────────────────────────────────────────────
tab_comp, tab_feat, tab_price, tab_compl, tab_gaps, tab_trace, tab_report = st.tabs(
    [
        "🏢 Competitors",
        "🔬 Features",
        "💰 Pricing",
        "📣 Complaints",
        "🚀 Gaps",
        "🕵️ Agent trace",
        "📄 Report",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Competitors
# ══════════════════════════════════════════════════════════════════════════════
with tab_comp:
    st.markdown("### Discovered competitors")
    if not competitors:
        st.info("No competitors found in this run.")
    else:
        try:
            import plotly.graph_objects as go

            names = [c["name"] for c in competitors]
            mentions = [c.get("mention_count", 0) for c in competitors]
            fig = go.Figure(
                go.Bar(
                    x=names,
                    y=mentions,
                    marker=dict(
                        color=mentions,
                        colorscale=[[0, "#1f6feb"], [1, "#58a6ff"]],
                        showscale=False,
                    ),
                    text=mentions,
                    textposition="outside",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=20, b=40, l=20, r=20),
                height=320,
                yaxis_title="Mention count",
                font=dict(family="Inter", color="#c9d1d9"),
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            pass

        st.markdown("#### Competitor details")
        for c in competitors:
            with st.expander(f"**{c['name']}** — {c.get('one_liner', '')[:90]}"):
                ca, cb = st.columns(2)
                ca.markdown(f"🌐 [{c['website']}]({c['website']})")
                if c.get("pricing_url"):
                    cb.markdown(f"💲 [Pricing page]({c['pricing_url']})")
                st.caption(f"Mentioned in {c.get('mention_count', 0)} discovery sources")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Features
# ══════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.markdown("### Feature matrix")
    st.caption(
        "✓ offered · ~ partial · ✗ not offered · ? unknown. "
        "Hover a mark for its quote; click it to open the page."
    )

    if not features:
        st.info("No feature data in this run.")
    else:
        feat_names = list(dict.fromkeys(f["feature"] for f in features))
        comp_names = [c["name"] for c in competitors] or list(
            dict.fromkeys(f["company"] for f in features)
        )
        cell_map = {(f["company"], f["feature"]): f for f in features}

        header_html = "<th style='text-align:left;padding:8px 12px'>Feature</th>" + "".join(
            f"<th style='padding:8px 12px'>{esc(n)}</th>" for n in comp_names
        )
        rows_html = ""
        for feat in feat_names:
            cells = ""
            for comp in comp_names:
                cell = cell_map.get((comp, feat))
                st_v = cell.get("status", "unknown") if cell else "unknown"
                icon = STATUS_ICON.get(st_v, "?")
                cls = STATUS_CLASS.get(st_v, "cell-unknown")
                tip = (
                    f'title="{esc(short(cell.get("quote") or "", 120))}"'
                    if cell and cell.get("quote")
                    else ""
                )
                if cell and cell.get("url") and st_v in ("yes", "partial", "no"):
                    icon = f'<a href="{esc(cell["url"])}" target="_blank" class="{cls}">{icon}</a>'
                cells += (
                    f'<td class="{cls}" {tip}'
                    f' style="text-align:center;padding:6px 12px">{icon}</td>'
                )
            rows_html += (
                f"<tr><td style='padding:6px 12px;color:#c9d1d9'>"
                f"<strong>{esc(feat)}</strong></td>{cells}</tr>"
            )

        _table_style = (
            "border-collapse:collapse;width:100%;background:#161b22;"
            "border-radius:10px;overflow:hidden;margin-bottom:1rem"
        )
        st.markdown(
            f'<div style="overflow-x:auto">'
            f'<table style="{_table_style}">'
            f'<thead style="background:#21262d;color:#c9d1d9">{header_html}</thead>'
            f'<tbody style="color:#c9d1d9">{rows_html}</tbody>'
            f"</table></div>",
            unsafe_allow_html=True,
        )

        try:
            import plotly.graph_objects as go

            cov: dict[str, dict[str, int]] = {}
            for feat in feat_names:
                cnts = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
                for comp in comp_names:
                    cell = cell_map.get((comp, feat))
                    sv = cell.get("status", "unknown") if cell else "unknown"
                    cnts[sv] = cnts.get(sv, 0) + 1
                cov[feat] = cnts
            colors_f = {
                "yes": "#3fb950",
                "partial": "#d29922",
                "no": "#f85149",
                "unknown": "#6e7681",
            }
            labels_f = {
                "yes": "✓ Yes",
                "partial": "~ Partial",
                "no": "✗ No",
                "unknown": "? Unknown",
            }
            fig2 = go.Figure()
            for sv in ("yes", "partial", "no", "unknown"):
                fig2.add_trace(
                    go.Bar(
                        name=labels_f[sv],
                        x=feat_names,
                        y=[cov[f][sv] for f in feat_names],
                        marker_color=colors_f[sv],
                    )
                )
            fig2.update_layout(
                barmode="stack",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=20, b=120, l=20, r=20),
                height=360,
                xaxis_tickangle=-35,
                font=dict(family="Inter", color="#c9d1d9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.markdown("#### Feature coverage by competitor count")
            st.plotly_chart(fig2, width="stretch")
        except ImportError:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Pricing
# ══════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.markdown("### Pricing comparison")
    st.caption("Per seat per month (USD) · annual price in brackets")

    if not pricing:
        st.info("No pricing data in this run.")
    else:
        price_map: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for tier in pricing:
            price_map[tier["company"]][tier["bucket"]].append(tier)

        comp_names_p = [c["name"] for c in competitors] or list(price_map.keys())
        header_p = "<th style='text-align:left;padding:8px 12px'>Company</th>" + "".join(
            f"<th style='padding:8px 12px'>{b.capitalize()}</th>" for b in BUCKET_ORDER
        )
        rows_p = ""
        for comp in comp_names_p:
            cells_p = ""
            for bucket in BUCKET_ORDER:
                tiers = price_map[comp].get(bucket, [])
                if not tiers:
                    cells_p += "<td style='color:#6e7681;text-align:center;padding:6px 12px'>—</td>"
                else:
                    txt = "<br>".join(esc(fmt_price(t)) for t in tiers)
                    cells_p += (
                        f"<td style='color:#c9d1d9;text-align:center;padding:6px 12px'>{txt}</td>"
                    )
            page = next((t["source_url"] for t in pricing if t["company"] == comp), None)
            rows_p += (
                f"<tr><td style='padding:6px 12px'>"
                f"<strong style='color:#e6edf3'>{esc(comp)}</strong>{source_link(page)}"
                f"</td>{cells_p}</tr>"
            )

        _table_style_p = (
            "border-collapse:collapse;width:100%;background:#161b22;"
            "border-radius:10px;overflow:hidden;margin-bottom:1rem"
        )
        st.markdown(
            f'<div style="overflow-x:auto">'
            f'<table style="{_table_style_p}">'
            f'<thead style="background:#21262d;color:#c9d1d9">{header_p}</thead>'
            f"<tbody>{rows_p}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

        try:
            import plotly.express as px

            rows_s = []
            for t in pricing:
                price = t.get("monthly_usd") or t.get("annual_monthly_usd")
                if price is not None and price > 0:
                    rows_s.append(
                        {
                            "company": t["company"],
                            "bucket": t["bucket"],
                            "price": price,
                            "tier_name": t.get("tier", t["bucket"]),
                        }
                    )
            if rows_s:
                fig3 = px.strip(
                    rows_s,
                    x="bucket",
                    y="price",
                    color="company",
                    hover_data=["tier_name", "company"],
                    category_orders={"bucket": BUCKET_ORDER},
                    labels={"price": "$/mo", "bucket": "Tier bucket"},
                    template="plotly_dark",
                )
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(22,27,34,1)",
                    margin=dict(t=30, b=40, l=20, r=20),
                    height=350,
                    font=dict(family="Inter", color="#c9d1d9"),
                )
                st.markdown("#### Price distribution by tier bucket")
                st.plotly_chart(fig3, width="stretch")
        except ImportError:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Complaints
# ══════════════════════════════════════════════════════════════════════════════
with tab_compl:
    st.markdown("### Complaint clusters")

    if not clusters:
        st.info("No complaint clusters in this run.")
    else:
        try:
            import plotly.graph_objects as go

            top_n = min(10, len(clusters))
            top_cl = sorted(clusters, key=lambda c: c["share"], reverse=True)[:top_n]
            fig4 = go.Figure(
                go.Pie(
                    labels=[c["label"] for c in top_cl],
                    values=[c["share"] for c in top_cl],
                    hole=0.55,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    marker=dict(
                        colors=[f"hsl({int(i * 360 / top_n)}, 60%, 50%)" for i in range(top_n)]
                    ),
                )
            )
            fig4.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=20, r=20),
                height=380,
                showlegend=False,
                font=dict(family="Inter", color="#c9d1d9"),
            )
            st.plotly_chart(fig4, width="stretch")
        except ImportError:
            pass

        st.markdown("#### Cluster details")
        for cl in sorted(clusters, key=lambda c: c["share"], reverse=True):
            with st.expander(
                f"**{cl['label']}** — {cl['share']:.0%} of complaints "
                f"· sev {cl['mean_severity']:.1f}/3 · {cl['distinct_docs']} docs"
            ):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Complaints", len(cl["complaint_ids"]))
                sc2.metric("Distinct docs", cl["distinct_docs"])
                sc3.metric("Mean severity", f"{cl['mean_severity']:.1f}")
                src_labels = " · ".join(SOURCE_LABEL.get(s, s) for s in cl["source_types"])
                st.caption("Sources: " + src_labels)
                if cl.get("companies"):
                    st.caption("Companies: " + ", ".join(cl["companies"]))
                shown = 0
                for cid in cl["complaint_ids"]:
                    if shown >= 3:
                        break
                    comp = complaint_by_id.get(cid)
                    if comp:
                        src_type = comp.get("source_type", "")
                        src = SOURCE_LABEL.get(src_type, src_type)
                        who = f"{comp.get('company', '')} · " if comp.get("company") else ""
                        st.markdown(
                            f'<div class="quote-block">"{esc(short(comp["quote"], 200))}"'
                            f'<br><small style="color:#6e7681">{esc(who)}{esc(src)}'
                            f"{source_link(comp.get('url'))}</small></div>",
                            unsafe_allow_html=True,
                        )
                        shown += 1

        if complaints:
            try:
                import plotly.express as px

                sev_counts = {1: 0, 2: 0, 3: 0}
                for c in complaints:
                    sev_counts[c.get("severity", 1)] += 1
                fig5 = px.bar(
                    x=["1 – Annoyance", "2 – Blocks workflow", "3 – Deal-breaker"],
                    y=[sev_counts[1], sev_counts[2], sev_counts[3]],
                    color_discrete_sequence=["#d29922", "#f85149", "#ff7b72"],
                    labels={"x": "Severity", "y": "Complaints"},
                    template="plotly_dark",
                )
                fig5.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(22,27,34,1)",
                    margin=dict(t=30, b=40, l=20, r=20),
                    height=280,
                    font=dict(family="Inter", color="#c9d1d9"),
                    showlegend=False,
                )
                st.markdown("#### Complaint severity distribution")
                st.plotly_chart(fig5, width="stretch")
            except ImportError:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Gaps
# ══════════════════════════════════════════════════════════════════════════════
with tab_gaps:
    st.markdown("### Ranked market gaps")

    if not opportunities:
        st.info("No ranked opportunities in this run. Check weak signals below.")
    else:
        try:
            import plotly.graph_objects as go

            opp_names = [o["title"][:40] for o in opportunities]
            opp_scores = [o["score"] for o in opportunities]
            fig6 = go.Figure(
                go.Bar(
                    x=opp_scores,
                    y=opp_names,
                    orientation="h",
                    marker=dict(
                        color=opp_scores,
                        colorscale=[[0, "#1f6feb"], [1, "#3fb950"]],
                        showscale=False,
                    ),
                    text=[f"{s:.1f}" for s in opp_scores],
                    textposition="outside",
                )
            )
            fig6.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=10, b=20, l=20, r=60),
                height=max(200, len(opportunities) * 48),
                xaxis=dict(range=[0, 10.5], title="Score (0–10)"),
                yaxis=dict(autorange="reversed"),
                font=dict(family="Inter", color="#c9d1d9"),
            )
            st.plotly_chart(fig6, width="stretch")
        except ImportError:
            pass

        ver = stats.get("verification") or {}
        if ver:
            st.caption(
                f"Verification: {ver.get('opportunities_verified', 0)}/"
                f"{ver.get('opportunities', 0)} gaps verified · {ver.get('quotes_passed', 0)}/"
                f"{ver.get('quotes_checked', 0)} quotes passed the verbatim check and the judge"
            )
        for i, opp in enumerate(opportunities, 1):
            cl = cluster_by_id.get(opp["cluster_id"], {})
            kind = opp.get("kind", "")
            kind_b = (
                badge("unmet need", "badge-unmet")
                if kind == "unmet_need"
                else badge("poor execution", "badge-poor")
            )
            v_b = (
                badge("✓ verified", "badge-verified")
                if opp.get("verified")
                else badge("unverified", "badge-unverified")
            )
            bd_str = " · ".join(f"{k} {v:.2f}" for k, v in opp.get("score_breakdown", {}).items())
            quotes = ""
            for eid in opp.get("evidence", [])[:5]:
                ev = complaint_by_id.get(eid)
                if ev:
                    ev_src_type = ev.get("source_type", "")
                    src = SOURCE_LABEL.get(ev_src_type, ev_src_type)
                    who = f"{ev.get('company', '')} · " if ev.get("company") else ""
                    quotes += (
                        f'<div class="quote-block">"{esc(short(ev["quote"], 220))}"'
                        f'<br><small style="color:#6e7681">{esc(who)}{esc(src)}'
                        f"{source_link(ev.get('url'))}</small></div>"
                    )
            n_comp = len(competitors) or 1
            st.markdown(
                f"""<div class="opp-card">
<div class="opp-title">{i}. {esc(opp["title"])}</div>
<div class="opp-score">{opp["score"]:.1f} / 10</div>
{kind_b} {v_b}<br><br>
<span class="stat-pill">📊 {cl.get("share", 0):.0%} of complaints</span>
<span class="stat-pill">📄 {cl.get("distinct_docs", 0)} documents</span>
<span class="stat-pill">⚠️ severity {cl.get("mean_severity", 0):.1f}/3</span>
<span class="stat-pill">🏢 {opp["coverage"] * n_comp:g}/{n_comp} competitors claim it</span>
<br><small style="color:#8b949e;font-size:.78rem">{esc(bd_str)}</small>
<p style="color:#c9d1d9;font-size:.9rem;margin-top:.6rem">{esc(opp.get("rationale", ""))}</p>
{quotes}
</div>""",
                unsafe_allow_html=True,
            )

    if weak_signals:
        st.markdown("---")
        st.markdown("#### ⚠️ Weak signals (didn't pass the evidence gates)")
        for ws in weak_signals:
            why = failed_gates(cluster_by_id.get(ws["cluster_id"], {}))
            st.markdown(
                f'<span class="stat-pill">{esc(ws["title"])} — {ws["score"]:.1f}/10'
                f"{' · missing ' + esc(why) if why else ''}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("")

    if opportunities:
        try:
            import plotly.graph_objects as go

            comp_keys = ["F", "S", "G", "E"]
            labels_c = {
                "F": "Frequency",
                "S": "Severity",
                "G": "Competitor gap",
                "E": "Evidence strength",
            }
            colors_c = ["#388bfd", "#d29922", "#3fb950", "#bc8cff"]
            fig7 = go.Figure()
            for ck, col in zip(comp_keys, colors_c, strict=False):
                fig7.add_trace(
                    go.Bar(
                        name=labels_c[ck],
                        x=[o["title"][:30] for o in opportunities],
                        y=[o.get("score_breakdown", {}).get(ck, 0) for o in opportunities],
                        marker_color=col,
                    )
                )
            fig7.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=20, b=100, l=20, r=20),
                height=380,
                xaxis_tickangle=-30,
                yaxis=dict(range=[0, 1.05], title="Component score (0–1)"),
                font=dict(family="Inter", color="#c9d1d9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.markdown("#### Score component breakdown")
            st.plotly_chart(fig7, width="stretch")
        except ImportError:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – Trace
# ══════════════════════════════════════════════════════════════════════════════
with tab_trace:
    st.markdown("### Agent trace")

    trace = load_trace(selected_path)
    if not trace:
        st.info(
            "No trace for this run: it was made with the linear pipeline "
            "(`mia run --linear`). Agent runs (`mia run`) write `trace.jsonl`."
        )
    else:
        researched = (stats.get("agent") or {}).get("researched") or []
        a1, a2, a3, a4 = st.columns(4)
        agent_stats = stats.get("agent") or {}
        a1.metric("Graph steps", max((r.get("step") or 0) for r in trace))
        a2.metric("Discovery rounds", agent_stats.get("discovery_rounds", "—"))
        a3.metric("Research rounds", max(agent_stats.get("verify_rounds", 1) - 1, 0))
        a4.metric("LLM requests in trace", sum(r.get("requests", 0) for r in trace))
        for line in researched:
            st.markdown(
                f'<div class="quote-block" style="font-style:normal">🔁 {esc(line)}</div>',
                unsafe_allow_html=True,
            )

        diagram = selected_path.parent / "graph.mmd"
        if diagram.exists():
            with st.expander("Graph diagram"):
                # the diagram is written by our own graph code (graph.mmd) and escaped
                st.iframe(
                    f'<div class="mermaid">{esc(diagram.read_text(encoding="utf-8"))}</div>'
                    f'<script src="{MERMAID_JS}"></script>'
                    "<script>mermaid.initialize({startOnLoad: true, theme: 'dark'});</script>",
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
                + (f" · {len(rows)} actions in parallel" if parallel else "")
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

        try:
            import plotly.express as px

            per_agent: dict[str, int] = defaultdict(int)
            for r in trace:
                per_agent[r.get("agent", "—")] += r.get("requests", 0)
            fig8 = px.bar(
                x=list(per_agent),
                y=list(per_agent.values()),
                labels={"x": "Agent", "y": "LLM requests"},
                template="plotly_dark",
            )
            fig8.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=20, b=40, l=20, r=20),
                height=280,
                font=dict(family="Inter", color="#c9d1d9"),
            )
            st.markdown("#### LLM requests per agent")
            st.plotly_chart(fig8, width="stretch")
        except ImportError:
            pass

        st.caption(f"Total trace entries: {len(trace)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – Report
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    report_md = selected_path.parent / "report.md"
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8")
        st.download_button("Download report.md", text, file_name="report.md")
        st.markdown(text)  # rendered without unsafe HTML
    else:
        st.info("No report.md next to this run.")
