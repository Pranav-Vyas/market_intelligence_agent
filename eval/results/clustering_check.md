# Clustering: hand check (task B9)

Run of 2026-10-17 on the demo market, `cluster_distance = 0.3` (average linkage, cosine distance
between complaint summaries): 229 complaints → 46 clusters of 2 or more, of which the 20 largest
are kept; 10 have at least 5 complaints. Three clusters were read complaint by complaint. Each
clause below is a complaint's normalized summary.

| Cluster (label) | Size | Belongs | Doesn't |
|---|---|---|---|
| recording meetings without consent or notice | 6 | 5: no pre-meeting notice, surreptitious recording, training on recordings without permission, recording without consent (2) | "subjectively discarded test recording" |
| unexpected charges and difficult cancellation | 6 | 5: auto-recurring charges, double charges, unauthorized charges, charges after cancelling, charged for unrequested features | "sudden usage limit enforcement" |
| unreliable meeting recording and joining | 29 | First 8 read: 5 about the bot joining meetings it wasn't invited to, or failing to join | "missing calendar view", "spamming contacts with signup emails", "stops recording mid meeting" |

**Verdict:** the first two are coherent. The largest cluster is too broad: average linkage chains
"bot joins meetings" with general reliability complaints, and its label reflects that.

Other thresholds (same summaries): 0.25 leaves 76 complaints alone and only 7 clusters reach 5
complaints; 0.35 merges distinct problems into clusters of 26–29. Try complete linkage or a
smaller threshold for the largest clusters when tuning (task B13).
