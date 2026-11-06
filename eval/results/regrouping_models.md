# Which model should regroup complaints? (task B13)

Run on 2026-11-05 with `eval/compare_regrouping.py`: three tries per model, scored against the
AI-drafted themes. B-cubed F1, with word-similarity seeds and no regrouping as the baseline.

| Complaints | Seeds alone | gemini-3.5-flash-lite (low thinking) | gemini-3.6-flash (medium thinking) | gemini-3.8-flash (medium thinking) |
|---|---|---|---|---|
| 243 (Day 6 run) | 0.461 | 0.593, 0.577, 0.614 | 0.545, 0.526, 0.621 | no answer in 3 tries (503) |
| 229 (Day 7 run) | 0.439 | 0.517, 0.519, one 503 | 0.533, 0.517, 0.517 | hung with no answer; stopped after an hour |
| Mean F1 | | 0.595 and 0.518 | 0.564 and 0.522 | none |
| Precision | | 0.75–0.87 | 0.83–0.95 | none |
| Recall | | 0.37–0.51 | 0.36–0.49 | none |
| Groups | | 12–25 | 26–30 | none |
| Output tokens per answer | | 900–1,500 | 6,200–7,800 | none |

**Decision: regrouping stays on flash-lite.** 3.6 makes cleaner groups but splits them more, so its
F1 is no better on three tries, and small groups miss the evidence gates. Three tries per model is a
small sample, and the themes are AI-drafted.

**Smart model: gemini-3.6-flash.** It answered every request that day, while 3.8 answered none. Both
allow about 20 requests a day, which is enough for the few calls that still use the smart model.

## Backup model: gemini-3.1-flash-lite

Run on 2026-11-06 with the same complaints, three tries each, both models at low thinking.

| Complaints | gemini-3.5-flash-lite | gemini-3.1-flash-lite |
|---|---|---|
| 243 (Day 6 run) | 0.583, 0.585, 0.585 | 0.624, 0.652, 0.574 |
| 229 (Day 7 run) | 0.565, 0.501, 0.547 | 0.526, 0.501, 0.592 |
| Purity of clusters with 5 or more complaints | 0.72–0.81 | 0.57–0.74 |
| Complaints in those clusters | 122–178 | 160–227 |

3.1 places more complaints but mixes more problems in a group, so its F1 is about the same. That's
good enough for a backup, not a reason to switch. It answered 40 of 40 tiny requests with no 503s
and allows 15 requests a minute; its daily limit wasn't reached.

