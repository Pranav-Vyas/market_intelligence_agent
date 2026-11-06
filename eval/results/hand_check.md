# Hand checks (tasks B14, B15)

Items are picked by `eval/hand_check.py` with a fixed seed. **The verdicts are AI-drafted, not yet
confirmed by a person:** read the sheet and overrule anything you disagree with.

| Market | Check | Right | Checked | Rate | Aim |
|---|---|---|---|---|---|
| AI calendar / scheduling assistants | clustering | 17 | 20 | 85% | 80% |
| AI meeting assistants | pricing | 20 | 23 | 87% | 90% |
| AI meeting assistants | features | 29 | 30 | 97% | 80% |
| AI meeting assistants | clustering | 19 | 20 | 95% | 80% |

## AI calendar / scheduling assistants (run 20261107-000551)

**clustering, not right:**

- 2 app is buggy for changing timings: about editing times in the app, not calendar sync
- 4 new app version is buggy: the quote is too vague to be a crash
- 12 cannot edit meeting notes in app: a missing edit feature, not a tracking error


## AI meeting assistants (run 20261105-184252)

**pricing, not right:**

- Otter|Pro|pro: annual price 8.49 is a promotional offer on the monthly plan; the annual price on the page is 8.33
- Otter|Business|business: annual price 24 is a 3-month promotional price; the annual price on the page is 19.99
- Avoma|Organization|business: annual price 24 is a discounted figure; the plan comparison on the page says 29

**features, not right:**

- tl;dv|Automated follow-up email drafting: should be yes, its homepage says its AI drafts follow-up emails

**clustering, not right:**

- 6 does not allow logging into google from iphone: a login failure, not a forced login

