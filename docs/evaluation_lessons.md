# Engineering Lessons: Building and Evaluating an LLM Styling System

Lessons from building ClosetGraph's evaluation (Stage 2) and tool-context work
(Stage 3) that generalize beyond this project - evaluation methodology and
architecture decisions alike. Each is grounded in the actual implementation and
observed behavior, not hypotheticals.

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

## 6. Reproducibility is not the same as coverage - do not seed to pass

**Issue.** Bounding the candidate pool for the token budget (a sampled subset of
all possible outfits) meant some golden checks referenced outfits the model was
never shown. The first instinct - and what we initially did - was to pick the
random seed so the referenced combinations landed in the sample.

**Root cause.** That conflates two different goals. Seeding for reproducibility
(stable runs) is good. Seeding so the eval passes is gaming: it hides sampling
coverage, and it silently breaks the moment the wardrobe changes.

**Fix.** The seed is now arbitrary and documented as reproducibility-only. A
check whose referenced item/combination is absent from the pool is reported
"inapplicable" rather than failed - distinguishing a missing item from a missing
combination (the items can all exist yet never co-occur in one sampled outfit).
Inapplicable checks are excluded from the pass rate, and applicable-coverage
(applicable / total checks) is reported so that shrinking the pool shows up as a
coverage problem even when nothing fails.

**Takeaway.** Separate "is the run reproducible" from "did the eval actually get
to test this." Never tune randomness to make checks pass; instead measure
coverage as a first-class metric, and treat "the model never had the chance" as
a distinct outcome from pass or fail.

## 7. Deterministic control flow for data; the model only for judgment

**Issue.** The first plan for "use weather when relevant" was an LLM router that
decided whether weather was relevant to the occasion.

**Root cause.** That made a routing component answer a styling question. Whether
weather *changes the outfit* is compositional reasoning - the stylist's job.
Worse, it added an LLM call (cost + the temp-0 nondeterminism of lesson 4) to a
decision that is really just "do we have weather data?".

**Fix.** Two separate decisions. A deterministic gate asks only "do we have or
can we fetch weather?" (is there an explicit input or a location). If yes, the
weather is resolved and passed to the stylist; the stylist then judges whether
and how much it matters for this occasion, told explicitly that weather
influences but never overrides the occasion's formality. The graph stays linear;
no router, no extra LLM call.

**Takeaway.** Split data-availability (deterministic, cheap, reproducible) from
judgment (the model's job). Do not spend an LLM call - and import its
nondeterminism - to decide something code can decide, and do not let a router
pre-empt the reasoning the downstream model exists to do.

## 8. Treat partial structured input as partial

**Issue.** Weather resolution was all-or-nothing: if the user supplied any field
(say temperature), the lookup was skipped entirely, so "3C in NYC" kept the
temperature but never discovered it was snowing.

**Root cause.** Presence of one field was treated as presence of all. User
precedence was applied at the whole-object level instead of per field.

**Fix.** Resolve field by field: user-provided fields always win, and the
fetcher is consulted only to fill the gaps (and only when a location is
available). Provenance became per-field (a sources map), so a context can be
part user-provided and part fetched, and a failed fill keeps the user's fields
rather than discarding everything.

**Takeaway.** When inputs are structured and partial, merge at the field level
with per-field provenance. "The user gave us something" rarely means "the user
gave us everything."

## 9. A controlled test must vary one variable at a time

**Issue.** Weather pairwise cases meant to prove "a rain jacket wins only in the
rain" and "a sweater wins only in wind" failed: the model kept the extra layer
even in the dry/calm control.

**Root cause.** The cases changed the condition (rainy->dry, windy->calm) but
held temperature at 12-14C - cool enough that a jacket or sweater is reasonable
regardless of rain or wind. Temperature and condition were confounded, so the
"drop the layer" expectation was not defensible and the test could not attribute
the result to the condition. The cases that used a real temperature gap (2C vs
26C) passed cleanly.

**Fix.** Decouple the variables: to isolate "condition", make the contrast
warm+rainy vs warm+dry so temperature does not independently justify the layer.
Correcting a confounded experiment is legitimate; relaxing the check to force a
pass would not be.

**Takeaway.** In a controlled comparison, change exactly one variable and hold
the rest at a level that does not independently explain the outcome. A failing
controlled test is as likely to indict the experiment's design as the system
under test - check the design before blaming the model.
