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
