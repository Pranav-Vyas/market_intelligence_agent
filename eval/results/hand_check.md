# Hand checks (tasks B14, B15)

Items are picked by `eval/hand_check.py` with a fixed seed. **The verdicts are AI-drafted, not yet
confirmed by a person:** read the sheet and overrule anything you disagree with.

| Market | Check | Right | Checked | Rate | Aim |
|---|---|---|---|---|---|
| AI calendar / scheduling assistants | clustering | 14 | 20 | 70% | 80% |
| AI meeting assistants | pricing | 21 | 23 | 91% | 90% |
| AI meeting assistants | features | 29 | 30 | 97% | 80% |
| AI meeting assistants | clustering | 19 | 20 | 95% | 80% |

## AI calendar / scheduling assistants (run 20261109-204210)

**clustering, not right:**

- 3 cannot edit meeting notes in app: a missing edit feature, not a meeting error
- 7 removed ability to change one off dates: a removed feature, not a date bug
- 9 missing feature to create events via ai: a feature request, not a failure to create events
- 11 forces user to link external calendar: about being forced to link a calendar, not a missing integration
- 15 poor ai scheduler: the AI scheduler works poorly, not a missing iOS feature
- 17 need to manually mark tasks as done: a missing automation, not glitchy task creation


## AI meeting assistants (run 20261109-204152)

**pricing, not right:**

- Otter|Business|business: the promotional annual price is now dropped, but the regular one (19.99) is missing
- Avoma|Organization|business: annual price 24 is a sale price; the plan comparison on the page says 29

**features, not right:**

- tl;dv|Automated follow-up email drafting: should be yes, its homepage says its AI drafts follow-up emails

**clustering, not right:**

- 6 does not allow logging into google from iphone: a login failure, not a forced login

