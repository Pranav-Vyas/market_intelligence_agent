# Clustering tuning (task B13): 229 complaints, run 20261105-184252

| Method | Threshold | B³ precision | B³ recall | B³ F1 | Clusters ≥5 | Complaints in them | Their purity | Largest |
|---|---|---|---|---|---|---|---|---|
| average | 0.2 | 0.929 | 0.178 | 0.299 | 2 | 16 | 1.0 | 9 |
| average | 0.25 | 0.794 | 0.226 | 0.351 | 7 | 57 | 0.624 | 13 |
| average | 0.3 | 0.675 | 0.325 | 0.439 | 10 | 103 | 0.739 | 29 |
| average | 0.35 | 0.489 | 0.365 | 0.418 | 15 | 163 | 0.621 | 29 |
| average | 0.4 | 0.303 | 0.517 | 0.382 | 12 | 214 | 0.44 | 60 |
| average | 0.45 | 0.121 | 0.785 | 0.209 | 3 | 227 | 0.22 | 146 |
| average | 0.5 | 0.061 | 0.993 | 0.114 | 1 | 228 | 0.123 | 228 |
| complete | 0.2 | 0.934 | 0.158 | 0.271 | 2 | 12 | 1.0 | 7 |
| complete | 0.25 | 0.83 | 0.185 | 0.303 | 2 | 15 | 1.0 | 9 |
| complete | 0.3 | 0.747 | 0.214 | 0.333 | 6 | 38 | 0.706 | 9 |
| complete | 0.35 | 0.675 | 0.257 | 0.373 | 9 | 65 | 0.66 | 13 |
| complete | 0.4 | 0.534 | 0.307 | 0.39 | 14 | 127 | 0.593 | 25 |
| complete | 0.45 | 0.418 | 0.346 | 0.378 | 24 | 206 | 0.575 | 26 |
| complete | 0.5 | 0.298 | 0.398 | 0.341 | 17 | 221 | 0.46 | 40 |
| average + LLM refinement | 0.3 | 0.838 | 0.348 | 0.492 | 13 | 102 | 0.778 | 18 |

Failure/need category agrees with the drafted themes on 0.904 of the complaints the refinement kept.

## Market gaps under other score weights

| Setting | Ranked, best first |
|---|---|
| default (F .35 S .25 G .25 E .15) | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Difficult subscription cancellation and high prices |
| more gap (F .25 S .25 G .35 E .15) | Poor transcription quality and inaccurate summaries; Security concerns regarding cloud storage and data handling; Difficult subscription cancellation and high prices |
| more frequency (F .45 S .2 G .2 E .15) | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Difficult subscription cancellation and high prices |
| more severity (F .25 S .35 G .25 E .15) | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Difficult subscription cancellation and high prices |
| equal (.25 each) | Poor transcription quality and inaccurate summaries; Security concerns regarding cloud storage and data handling; Difficult subscription cancellation and high prices |

## Market gaps under other evidence gates

| Setting | Ranked, best first |
|---|---|
| default (5 complaints, 3 docs, 2 companies, 2 sources) | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Difficult subscription cancellation and high prices |
| no company gate | Recording meetings without consent; Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Difficult subscription cancellation and high prices |
| no source gate | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Forced google or microsoft login; Forced calendar access and permission loops; Difficult subscription cancellation and high prices; Unwanted meeting bot joins unexpectedly |
| 2 sources or 3 companies | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries; Forced calendar access and permission loops; Difficult subscription cancellation and high prices |
| stricter (8 complaints, 5 docs) | Security concerns regarding cloud storage and data handling; Poor transcription quality and inaccurate summaries |
