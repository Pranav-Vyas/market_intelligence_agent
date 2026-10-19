"""Streamlit v1 – Market Gap Scout browser (task C11, Day 5).

Pick a run → tabs for Competitors / Features / Pricing / Complaints / Gaps / Agent Trace.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

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
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "data" / "runs"
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
    st.caption("Market Gap Scout · Day 5 UI (C11)")

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
tab_comp, tab_feat, tab_price, tab_compl, tab_gaps, tab_trace = st.tabs(
    ["🏢 Competitors", "🔬 Features", "💰 Pricing", "📣 Complaints", "🚀 Gaps", "🕵️ Trace"]
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
            st.plotly_chart(fig, use_container_width=True)
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
    st.caption("✓ offered · ~ partial · ✗ not offered · ? unknown")

    if not features:
        st.info("No feature data in this run.")
    else:
        feat_names = list(dict.fromkeys(f["feature"] for f in features))
        comp_names = [c["name"] for c in competitors] or list(
            dict.fromkeys(f["company"] for f in features)
        )
        cell_map = {(f["company"], f["feature"]): f for f in features}

        header_html = "<th style='text-align:left;padding:8px 12px'>Feature</th>" + "".join(
            f"<th style='padding:8px 12px'>{n}</th>" for n in comp_names
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
                    f'title="{short(cell.get("quote") or "", 120)}"'
                    if cell and cell.get("quote")
                    else ""
                )
                cells += (
                    f'<td class="{cls}" {tip}'
                    f' style="text-align:center;padding:6px 12px">{icon}</td>'
                )
            rows_html += (
                f"<tr><td style='padding:6px 12px;color:#c9d1d9'>"
                f"<strong>{feat}</strong></td>{cells}</tr>"
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
            st.plotly_chart(fig2, use_container_width=True)
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
                    txt = "<br>".join(fmt_price(t) for t in tiers)
                    cells_p += (
                        f"<td style='color:#c9d1d9;text-align:center;padding:6px 12px'>{txt}</td>"
                    )
            rows_p += (
                f"<tr><td style='padding:6px 12px'>"
                f"<strong style='color:#e6edf3'>{comp}</strong></td>{cells_p}</tr>"
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
                st.plotly_chart(fig3, use_container_width=True)
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
            st.plotly_chart(fig4, use_container_width=True)
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
                            f'<div class="quote-block">"{short(comp["quote"], 200)}"'
                            f'<br><small style="color:#6e7681">{who}{src}</small></div>',
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
                st.plotly_chart(fig5, use_container_width=True)
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
            st.plotly_chart(fig6, use_container_width=True)
        except ImportError:
            pass

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
            st.markdown(
                f"""<div class="opp-card">
<div class="opp-title">{i}. {opp["title"]}</div>
<div class="opp-score">{opp["score"]:.1f} / 10</div>
{kind_b} {v_b}<br><br>
<span class="stat-pill">📊 {cl.get("share", 0):.0%} of complaints</span>
<span class="stat-pill">⚠️ severity {cl.get("mean_severity", 0):.1f}/3</span>
<span class="stat-pill">🏢 coverage {opp["coverage"]:.0%}</span>
<br><small style="color:#8b949e;font-size:.78rem">{bd_str}</small>
<br><br><p style="color:#c9d1d9;font-size:.9rem">{opp.get("rationale", "")}</p>""",
                unsafe_allow_html=True,
            )
            shown = 0
            for eid in opp.get("evidence", []):
                if shown >= 3:
                    break
                ev = complaint_by_id.get(eid)
                if ev:
                    ev_src_type = ev.get("source_type", "")
                    src = SOURCE_LABEL.get(ev_src_type, ev_src_type)
                    who = f"{ev.get('company', '')} · " if ev.get("company") else ""
                    st.markdown(
                        f'<div class="quote-block">"{short(ev["quote"], 220)}"'
                        f'<br><small style="color:#6e7681">{who}{src}</small></div>',
                        unsafe_allow_html=True,
                    )
                    shown += 1
            st.markdown("</div>", unsafe_allow_html=True)

    if weak_signals:
        st.markdown("---")
        st.markdown("#### ⚠️ Weak signals (didn't pass evidence gates)")
        for ws in weak_signals:
            st.markdown(
                f'<span class="stat-pill">{ws["title"]} — {ws["score"]:.1f}/10</span>',
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
            st.plotly_chart(fig7, use_container_width=True)
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
            "No trace file found for this run.\n\n"
            "Traces are written by the LangGraph graph engine (`mia run --graph`)."
        )
    else:
        try:
            import plotly.express as px

            df_rows = [
                {
                    "step": row.get("step") or i,
                    "agent": row.get("agent", "—"),
                    "competitor": row.get("competitor") or "(all)",
                    "action": row.get("action", ""),
                    "summary": row.get("result_summary", ""),
                    "requests": row.get("requests", 0),
                    "tokens": row.get("tokens", 0),
                    "at": row.get("at", ""),
                }
                for i, row in enumerate(trace)
            ]
            fig8 = px.bar(
                df_rows,
                x="agent",
                y="tokens",
                color="agent",
                hover_data=["action", "competitor", "summary"],
                labels={"tokens": "Tokens used", "agent": "Agent"},
                template="plotly_dark",
            )
            fig8.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,1)",
                margin=dict(t=20, b=40, l=20, r=20),
                height=300,
                font=dict(family="Inter", color="#c9d1d9"),
                showlegend=False,
            )
            st.plotly_chart(fig8, use_container_width=True)
        except ImportError:
            pass

        agents = sorted({r.get("agent", "") for r in trace if r.get("agent")})
        sel_agent = st.selectbox("Filter by agent", ["All"] + agents)
        filtered = (
            trace if sel_agent == "All" else [r for r in trace if r.get("agent") == sel_agent]
        )

        for row in filtered:
            step_lbl = f"Step {row.get('step', '?')}" if row.get("step") is not None else ""
            comp_lbl = f" · {row['competitor']}" if row.get("competitor") else ""
            with st.expander(
                f"[{step_lbl}] **{row.get('agent', '')}{comp_lbl}** — {row.get('action', '')}  "
                f"·  {row.get('requests', 0)} req · {row.get('tokens', 0)} tok"
            ):
                st.markdown(f"**Result:** {row.get('result_summary', '')}")
                if row.get("at"):
                    st.caption(f"⏱ {row['at']}")

        st.caption(f"Total trace entries: {len(trace)}")
