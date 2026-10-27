# Engineering Lessons: Evaluating an LLM Styling System

Three lessons from building ClosetGraph's Stage 2 evaluation that generalize
beyond this project. Each is grounded in the actual implementation and observed
behavior, not hypotheticals.

## 1. A silent fallback is worse than a crash

**Issue.** The stylist wrapped its Groq call and JSON parsing in a broad
`except Exception` that, on any failure, returned a hardcoded keyword-based
ranker with three canned reasoning strings. For several occasions the demo
produced identical, generic reasoning across every outfit.

**Root cause.** The catch-all conflated two fundamentally different failures:
the model producing a *wrong* answer versus the model producing *no* answer. On
the Groq free tier the real failures were rate limits (HTTP 429) and the
reasoning model spending its entire output budget on reasoning tokens and
emitting empty content. The fallback turned all of these into plausible-looking
success, so they were invisible until inspected by hand.

**Fix.** Removed the fallback and the keyword ranker entirely; kept a single
stricter "JSON-only" retry; a persistent failure now raises
(`closetgraph/stylist.py`). The failures immediately became visible as real
exceptions and, in Langfuse, as traced errors.

**Takeaway.** In an LLM system, an error handler must never fabricate a
successful-looking result. Fabricated output silently corrupts every downstream
metric and hides operational problems (quota, auth, truncation). Fail loud, and
keep "the model was wrong" distinct from "the model never ran."

## 2. Uncontrolled upstream randomness invalidates the evaluation

**Issue.** Golden checks of the form "does the model rank combination X" passed
and failed non-reproducibly from run to run.

**Root cause.** Candidate generation drew 20 of ~210 outfit combinations with an
*unseeded* `random.sample`. The evaluation implicitly assumed the model was
choosing from a stable set, but the set was resampled every run: two
consecutive generations overlapped on only 4 of 20 outfits, and a specific
combination a check depended on had roughly a 40% chance of even being offered.
Those checks were testing whether a random sampler happened to include an
outfit, not whether the model reasoned correctly.

**Fix.** Seeded the sampler (`random.Random(1146)`), with the seed chosen so the
sample contains every combination the golden set references
(`closetgraph/candidates.py`). Runs became reproducible and the checks started
measuring the model instead of the sampler.

**Takeaway.** Any nondeterministic input upstream of an evaluation must be pinned
or the eval measures noise. Before trusting a result, ask what varies between
runs that is not the thing under test, and control it. (Note the model itself is
still nondeterministic even at temperature 0 - that residual noise is handled
separately with repeat runs, not pretended away.)

## 3. Test the relationship, not an absolute rank, for compositional reasoning

**Issue.** The original layering checks demanded a specific layered outfit (e.g.
tank + blazer) appear in the model's top 5. All four layering cases failed,
which read as "the model cannot reason about layering."

**Root cause.** Forcing one specific outfit into a top 5 among 20 competitors
conflates two different claims: "can the model judge this composition" and "does
the model prefer this exact outfit over nineteen others for this occasion." The
model declining to rank one combo first is not evidence it cannot reason about
that combo.

**Fix.** Added pairwise suitability tests (`compare_pairwise`): the model
compares two outfits directly - a base outfit versus the same outfit plus a
layer - for an occasion, and the context is then *reversed* (a professional
occasion versus a casual one) to check that the transformation is
context-dependent rather than a blanket rule. Observed: both the 20B and 120B
models chose the elevated outfit correctly in every favoring context, so the
capability was present all along; the 20B's only weakness was failing to
de-escalate (it kept a layer even for a nightclub), which the reversal test
isolated cleanly.

**Takeaway.** To evaluate compositional or relational reasoning, test the
relationship (A vs B) rather than an absolute outcome (B must rank first of N),
and include a reversal so you are measuring understanding of the transformation
rather than a memorized preference. "The model can reason" and "the model ranks
the way I expected" are different claims; rigid top-N checks quietly test the
second while you believe you are testing the first.

## 4. Temperature 0 does not mean reproducible

**Issue.** Identical evaluation inputs produced different rankings across runs; a
check could pass one run and fail the next with nothing changed.

**Root cause.** Temperature 0 makes token selection greedy but does not make a
hosted reasoning model deterministic - the reasoning trace, server-side
batching, and provider nondeterminism still vary the output. Observed: two
golden cases that were identical at the time (same occasion and check) returned
different results; and in the 20B-vs-120B comparison the 120B returned "B" then
"A" on the same pairwise input across two runs.

**Handling.** Treat single-run pass/fail as noisy. The diagnostic runner
(`run_pairwise_set`) repeats each case N times and reports how often the expected
relationship holds, so conclusions rest on a rate rather than one sample.

**Takeaway.** Do not assume temperature 0 gives reproducibility from a hosted
model. Measure run-to-run variance before trusting any single eval number, and
report pass rates over repeats rather than a lone pass/fail.

## 5. Match the verifier to the claim, and never let a model grade itself

**Issue.** The golden set mixes objective facts (ownership; "this item should not
top the ranking"), relational judgments (is B more suitable than A here), and
subjective qualities (is the palette cohesive). One check mechanism cannot serve
all three well.

**Root cause.** Different claim types need different verifiers. Using code for
everything cannot judge subjective quality; using an LLM for everything wastes
calls on facts code can check exactly and applies brittle exact-match to things
that are inherently a matter of degree.

**Fix.** Three mechanisms, routed by expectation type: deterministic Python for
hard constraints and ranking preferences, a pairwise stylist comparison for
relational claims, and an LLM-as-judge only for the subjective color cases. The
judge runs on a different model (120B) than the stylist (20B) so it is not
grading its own family's output.

**Takeaway.** Pick the cheapest verifier that fits the claim - code for
objective, pairwise for relational, an LLM judge for subjective - and never
evaluate a model with itself. Self-grading correlates errors and inflates
scores.
