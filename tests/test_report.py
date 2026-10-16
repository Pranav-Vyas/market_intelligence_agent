from __future__ import annotations

from mia.report.markdown import Citations, render
from mia.schemas import (
    Competitor,
    Complaint,
    ComplaintCluster,
    FeatureCell,
    MarketReport,
    Opportunity,
    PricingTier,
)


def _report(**overrides) -> MarketReport:
    complaint = Complaint(
        id="d1:0:0",
        company="Otter",
        summary="too expensive",
        quote="Way too pricey.",
        severity=2,
        chunk_id="d1:0",
        doc_id="d1",
        source_type="reddit",
        url="https://reddit.com/r/x/1",
    )
    base = dict(
        market="AI meeting assistants",
        competitors=[Competitor(name="Otter", website="https://otter.ai")],
        features=[
            FeatureCell(
                company="Otter", feature="Transcription", status="yes", url="https://otter.ai"
            )
        ],
        pricing=[
            PricingTier(
                company="Otter",
                tier="Basic",
                bucket="free",
                monthly_usd=0,
                source_url="https://otter.ai/pricing",
            ),
            PricingTier(
                company="Otter",
                tier="Pro",
                bucket="pro",
                monthly_usd=8.33,
                per_seat=True,
                source_url="https://otter.ai/pricing",
            ),
        ],
        complaints=[complaint],
        clusters=[
            ComplaintCluster(
                id="c0",
                label="too expensive",
                complaint_ids=["d1:0:0"],
                share=1.0,
                distinct_docs=1,
                source_types=["reddit"],
                mean_severity=2.0,
                companies=["Otter"],
            )
        ],
        opportunities=[
            Opportunity(
                title="Too expensive",
                cluster_id="c0",
                kind="unmet_need",
                coverage=0.0,
                score=8.4,
                score_breakdown={"F": 1.0},
                rationale="r",
                evidence=["d1:0:0"],
                verified=True,
            )
        ],
        weak_signals=[],
        stats={},
    )
    base.update(overrides)
    return MarketReport(**base)


def test_citations_number_in_order_of_first_use():
    c = Citations()
    assert c.ref("https://a") == "[1]"
    assert c.ref("https://b") == "[2]"
    assert c.ref("https://a") == "[1]"
    assert c.ref(None) == ""
    assert c.items() == [(1, "https://a"), (2, "https://b")]


def test_render_contains_data_and_citations():
    md = render(_report())
    assert "| Transcription | ✓[1] |" in md
    assert "Free" in md and "$8.33/seat/mo" in md
    assert '"Way too pricey."' in md
    assert "[3] https://reddit.com/r/x/1" in md
    assert "### 1. Too expensive (8.4 / 10)" in md


def test_render_without_opportunities_says_so():
    md = render(_report(opportunities=[]))
    assert "No pain point passed the evidence gates" in md


def test_render_warns_about_stubs():
    md = render(_report(stats={"stubs": ["discovery.discover_competitors"]}))
    assert "fake data" in md


def test_summary_names_top_complaints_and_gaps():
    md = render(_report())
    assert "## Summary" in md
    assert "**Most common complaints:** too expensive (100%)." in md
    assert "**Top market gaps:** Too expensive (8.4/10, unmet need)." in md


def test_gap_reads_like_the_topic_example():
    md = render(_report())
    assert "- **Frequency:** 100% of complaints (1 complaint, 1 document, Reddit)" in md
    assert "- **Coverage:** 0/1 competitors claim to address it. r" in md
    assert "- **Evidence** ✓ verified:" in md
    assert '"Way too pricey." (Otter, Reddit)' in md


def test_weak_signals_say_which_gate_failed():
    weak = _report().opportunities[0].model_copy(update={"verified": False})
    md = render(_report(opportunities=[], weak_signals=[weak]))
    assert "1 complaint (needs 5); 1 document (needs 3); only Reddit (needs 2 source types)" in md


def test_prices_show_the_annual_price_in_brackets():
    tier = PricingTier(
        company="Otter",
        tier="Pro",
        bucket="pro",
        monthly_usd=16.99,
        annual_monthly_usd=8.33,
        per_seat=True,
        source_url="https://otter.ai/pricing",
    )
    annual_only = tier.model_copy(update={"monthly_usd": None})
    md = render(_report(pricing=[tier]))
    assert "$16.99/seat/mo ($8.33)" in md
    assert "$8.33/seat/mo (annual)" in render(_report(pricing=[annual_only]))


def test_complaints_table_shows_an_example_quote():
    assert '| "Way too pricey." [3] |' in render(_report())


def test_method_reports_verification():
    stats = {
        "verification": {
            "opportunities": 2,
            "opportunities_verified": 1,
            "quotes_checked": 10,
            "quotes_passed": 8,
            "quote_pass_rate": 0.8,
        }
    }
    assert "- Verification: 1/2 ranked gaps verified; 8/10 quotes passed (80%)" in render(
        _report(stats=stats)
    )
