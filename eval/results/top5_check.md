# Top pain points: hand check (task B13)

Run `20261105-020724` on the demo market, 2026-11-05: 229 complaints, 3 ranked market gaps and
6 product failures. Each row was judged by reading the quotes and every complaint summary in the
cluster.

**Status: AI-drafted, not yet signed off.** The task is done when all three team members agree
that these look sensible, so read them at the Day 7 demo and overrule anything here.

| # | Pain point | Kind, score | Verdict | Why |
|---|---|---|---|---|
| 1 | Security concerns regarding cloud storage and data handling | poor execution, 6.8 | Sensible, broad | Real and across 5 vendors, but it mixes general worry about cloud storage with one vendor's incident (3 of 8 quotes are tl;dv's), and vendors' security pages are what make it "poor execution". |
| 2 | Poor transcription quality and inaccurate summaries | poor execution, 6.7 | Sensible | The clearest gap, and Topic.md's own example. Two of 8 quotes describe outright failures rather than poor quality. |
| 3 | Difficult subscription cancellation and high prices | unmet need, 5.6 | Partly | Two problems in one label. The price half is real, but "unmet need" is misleading: 6 of 8 vendors have a free plan. Coverage reads the cancellation half of the label and finds nothing. |
| 4 | Ignores security and data deletion requests | product failure, 7.5 | Sensible, narrow | Well evidenced, but 4 of 6 quotes are about Otter. |
| 5 | Audio upload and processing failures | product failure, 7.0 | Sensible | Coherent, across 5 vendors. App Store only, as expected for an upload bug. |

**Compared with Day 6.** The former top "gaps" (app crashes, recordings lost) are now product
failures, and the ranked gaps are ones a competing product could act on.

## Held back as weak signals, and worth discussing

- **Forced Google or Microsoft login (6.6):** a real opening, but only in App Store reviews, so it
  misses the two-source gate.
- **Forced calendar access (5.8):** the same, across 4 vendors. A rule of "2 sources or 3 companies"
  would rank it (see `clustering_tuning.md`).
- **Recording without consent (7.2):** every complaint is about Otter in this run, so the company
  gate holds it back. The fresh run also found Fireflies complaints and ranked it first.

## Known limits

- **Results vary between runs.** The regrouping is one LLM call, so small changes move complaints
  and rename clusters. Over prompt variations of the Day 6 complaints, B-cubed F1 ranged 0.52–0.59.
  The fresh run, on a newly collected corpus, ranked 8 market gaps rather than 3.
- **About 45% of complaints join no pain point** (125 of 229 are clustered here).
- **The themes behind the F1 numbers are AI-drafted** from the summaries alone.
- **Quote pass rate is under the 95% aim:** 90% here and 81% in the fresh run. Reading the 16
  rejected quotes in an earlier check, about 13 rejections were right, so the rate reflects quotes
  that don't fit their cluster, not a harsh judge. Only passing quotes reach the report.
