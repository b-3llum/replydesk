# evolve: self-improving reply prompt

Evolves the system prompt that writes owner replies to Google reviews, using parallel
mutation agents on Luminal, then checks whether the gain is real. Open `report.html` for the
full write-up and `summary.json` for the numbers.

## Results (run 20261011-170151, judge v2-strict-editor, DeepSeek-V4.1-Flash)

| | train (40) | held-out (30) | fresh (40) |
|---|---|---|---|
| v0 (one line) | 43.8 | 40.5 | 41.6 |
| rules (hand-written, not optimized) | | | 89.0 |
| round-1 best `p16cf58ceee` | 94.5 | | 91.5 |
| final best `p992e41d4d7` (round 5) | 96.3 | 94.0 | 92.6 |

Independent blind judge (Claude, pairwise, randomized A/B, no rubric) on the fresh set:
final best beat the rules prompt 31-9 (p = 0.0007); final best vs round-1 best was 18-22
(p = 0.64): the round-1 prompt won 12 of 14 complaints and 6 of 6 3-star reviews, the final
prompt won 16 of 20 4-5 star reviews. Most of the gain comes from writing the rules down;
the loop adds a real but smaller gain; rounds 2-5 overfit our scorer on complaints.

Both evolved prompts embed real training reviews (names, review text, phone numbers) in their
examples. Replace those with invented placeholders before using them in production.

## Data splits

- `split.json`: 40 training reviews (5 businesses) and 30 held-out reviews (5 other
  businesses), rating-only reviews excluded, fixed seed.
- `validation.json` → `fresh_review_ids`: 40 reviews from 22 businesses scraped after the run.
- Held-out was scored once, after the last round. Nothing from held-out or fresh reviews
  appears in any candidate prompt.

## Files

The repo is public, so only code and anonymized results are committed. Everything holding
review text, reviewer names or generated replies is git-ignored and stays on the machine that
ran it (the admin card and the scripts read it from disk).

Committed:

| Path | What |
|---|---|
| `*.py` | split, scorer, loop, validation, blind tally, report, export |
| `prompts/v0.txt`, `prompts/rules.txt` | one-line baseline and hand-written rules baseline |
| `split.json` | review ids of the training and held-out sets |
| `summary.json` | all scores and stats, numbers only |
| `report.html` | standalone report, reviewer names reduced to initials |

Local only (git-ignored):

| Path | What |
|---|---|
| `prompts/best.txt` | final best prompt (its examples quote real reviewers) |
| `runs/<run_id>/` | `state.json` (population, worst pairs) and every candidate prompt |
| `results.json` | the run, with held-out replies per review |
| `validation.json`, `blind_pairs.json` | fresh-set replies, blind pairs and votes |
| `exports/` | Mongo snapshot: reviews, businesses, `prompt_runs`, un-blinded judgments |
| `cache/`, `logs/` | generation/judge caches (reruns cost zero API calls) and per-call token log |
| `report_full.html` | the report with real names |

Runs `20261011-165054` and `20261011-165227` are step-2 attempts that failed on HTTP 429 before
request pacing was added; `20261011-165545` is the successful one-round, 4-agent step 2.

## Commands

```
python -m evolve.split                     # rebuild split.json (refuses to overwrite)
python -m evolve.score                     # score prompts/v0.txt on training
python -m evolve.run --rounds 5 --agents 8 --k 4 [--resume RUN_ID] [--stop-at HH:MM] [--yes]
python -m evolve.validate                  # fresh-set scores + blind pairs
python -m evolve.blind_tally votes.json    # un-blind judge votes into validation.json
python -m evolve.report                    # rebuild report.html (initials) and report_full.html
python -m evolve.export_data               # refresh exports/ from Mongo and summary.json
```

Luminal notes: `chat_template_kwargs: {"thinking": false}` is ignored by this endpoint; the
top-level `"thinking": {"type": "disabled"}` is what turns reasoning off (both are sent). The
rate limit is per API key at roughly 1 request/second sustained, shared with anything else on
the key; `EVOLVE_LLM_RPS` and `EVOLVE_LLM_CONCURRENCY` tune the client.
