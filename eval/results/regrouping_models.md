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
