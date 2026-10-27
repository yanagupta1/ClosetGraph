# Stage 2 Eval Results

First recorded run of the golden set against the live graph.

- Date: 2026-10-07
- Model: `openai/gpt-oss-20b` (Groq free tier), temperature 0, `max_completion_tokens` 8192
- Candidates: seeded sample (seed 1146), 20 of 210 Cartesian combos
- Retrieval: stratified, 2 chunks per category (occasion, layering, garments, color)
- Run: throttled 60s per entry to stay under the free-tier tokens-per-minute limit

## Metrics

- Hard checks: **15/21 passed (71%)**
- Judged color-cohesion cases: **3 pending** (LLM judge not implemented yet)
- Caveat: single run. `gpt-oss-20b` is **not deterministic at temperature 0** (see g05 vs g18 below), so hard-check pass/fail has run-to-run variance of roughly 1-2 checks.

## Passed (15)

g01, g03, g04, g06, g07, g08, g09, g10, g11, g15, g16, g17, g19, g20, g21

## Failures (6)

### Layering elevation not surfaced (4): g02, g12, g13, g14

Each expected a specific top+layer combo to appear in the ranked top 5:

| id | occasion | expected combo |
| --- | --- | --- |
| g02 | team happy hour, casual bar | t1 black tank + l1 blazer |
| g12 | work happy hour, relaxed-but-put-together | t3 turtleneck + l2 cardigan |
| g13 | dressing up a plain top for a client meeting | t4 navy long-sleeve + l1 blazer |
| g14 | dinner date, upscale restaurant | t5 sleeveless blouse + l1 blazer |

Diagnosis: **not** a retrieval or candidate-coverage problem. The relevant
layering guidance (Structured Layers / Soft Layers / Coverage Through Layering)
was retrieved for all four occasions, and the combos were present in the seeded
candidate pool. The model received the guidance and had the option available but
did not rank the layered combo into its top 5. This is model
application/judgment, or an over-strict expectation, not missing context.

### Denim preference (2): g05, g18

Both are identical entries: casual weekend brunch, expecting b2 (dark clean
jeans) ranked above b5 (relaxed denim).

- g05: b2 ranked 3, b5 ranked 1 (inverted)
- g18: b2 not ranked at all

Because g05 and g18 are identical checks, their divergent results are purely
model nondeterminism. One of the two is effectively redundant.

## Judged pending (3) - color cohesion, need LLM judge

- g22 team happy hour -> t4, b4, a1
- g23 dinner date -> t5, b1
- g24 brunch -> t1, b4

## Known limitations

- Single-run metrics; nondeterminism at temp 0 can flip ~1-2 checks per run.
- Retrieval and candidate coverage verified adequate; remaining failures are
  model reasoning or expectation calibration.
- LLM judge for the three color-cohesion cases not implemented.
- Latency and token usage per call are visible in Langfuse, not captured here.

## Pairwise A/B: gpt-oss-20b vs gpt-oss-120b (2026-10-20)

Ran the three pairwise compositional cases (favoring + reversal) twice per
model, throttled. "B favored" = model ranks the elevated/cleaner outfit first
for the favoring (professional/polished) occasion.

| case (favoring context) | 20b favors B | 120b favors B | 20b reversal | 120b reversal |
| --- | --- | --- | --- | --- |
| p01 blazer, client meeting | 2/2 | 2/2 | A, A | A, A |
| p02 clean denim, business-casual Friday | 2/2 | 2/2 | B, B | B, A |
| p03 cardigan, museum lunch | 2/2 | 2/2 | B, B | A, A |

Findings:

1. Both models get every favoring direction right (6/6), including the 20b.
   The earlier g02/g12/g13/g14 "failures" were a measurement artifact of
   forcing a specific combo into the top 5, not a reasoning failure. Asked the
   direct A-vs-B question, the capability is clearly present.
2. The models differ on the reversal (does elevation stop winning in a casual
   context). 120b de-escalates correctly - drops the cardigan for a nightclub
   (p03) - while 20b keeps favoring the more put-together option. 120b is
   better at "layering does not always help"; 20b has a mild always-elevate bias.
3. Nondeterminism confirmed at temp 0: 120b p02 reversal returned B then A on
   identical input.

Decision: stay on gpt-oss-20b (it nails the primary compositional judgment;
120b's only edge is reversal nuance, not worth the extra output tokens/latency
on the free tier). The deterministic formality engine is not needed - the
reasoning is present. Revisit 120b only if context-dependent de-escalation
becomes a priority.

## Color-cohesion LLM judge, first verdicts (2026-10-20)

Judge model gpt-oss-120b scoring the 20B stylist's top pick for g22-g24 against
the draft rubrics. Compared against a manual read, the judge was faithful to
each rubric.

| id | top pick | verdict | note |
| --- | --- | --- | --- |
| g22 | navy top + dark jeans + blazer | pass, 3/5 | cohesive blue tones + black anchor, but the reasoning never states the colour logic (fair dock) |
| g23 | cream blouse + black trousers | pass, 5/5 | classic elevated neutral contrast |
| g24 | tank + camel trousers + beige cardigan | fail, 1/5 | g24's rubric judges the REASONING for colour awareness; the reasoning mentions none, so the fail is rubric-faithful even though the palette itself is fine |

Findings:

1. Two different things are measured, and the split is intentional. g22 and g23
   judge OUTFIT COHESION (do the chosen colours actually work together); g24
   judges the COLOUR-AWARE EXPLANATION (does the reasoning articulate colour
   balance). g24 is therefore strict by design: a cohesive outfit still fails if
   the reasoning is colour-silent. This distinction is stated explicitly in
   g24's rubric.
2. The judge surfaced a stylist output inconsistency: g23's reasoning references
   a "black structured blazer" that is not in the ranked item list (t5, b1).
   The items and the prose disagree - a stylist quality issue, not a judge one.

## Committed eval-runner run (2026-10-27)

First run through the committed CLI
(`scripts/run_evals.py --suite both --repeats 1 --delay 60`); full report in
`docs/stage2-eval-report.json`.

- Pairwise: p01 (blazer) favored B; p02 (clean denim) favored B; p03 (soft
  cardigan) favored A - on this single run the cardigan did NOT beat the plain
  turtleneck + skirt for a museum lunch. Single run, so noisy (see lesson 4);
  p01/p02 match the earlier A/B.
- Golden: did not complete. The first stylist request hit Groq's free-tier
  per-request cap - HTTP 413, "Request too large ... TPM Limit 8000, Requested
  8175." A single 20-candidate golden prompt plus its completion allotment now
  exceeds 8000 tokens/minute, so throttling (spacing calls) cannot help - the
  request itself is too large. `run_golden_set` has no per-case catch, so the
  first 413 aborted the whole golden pass; the runner recorded it in `errors`
  and continued to pairwise instead of crashing.

The runner behaved correctly (ran pairwise, captured the golden error without
crashing). The earlier completed golden pass (15/21) still stands. Fix
(follow-up): bring the golden request under 8000 - lower max_completion_tokens
and/or compact the candidate serialization or retrieve fewer chunks.
