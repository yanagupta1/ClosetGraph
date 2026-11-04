# ClosetGraph

ClosetGraph is a wardrobe intelligence project that recommends complete outfits from a user's own closet for real occasions like client meetings, dinners, interviews, and casual events.

The project is being built as a notebook-first LangGraph system for the Hugging Face Agents Course final project. The goal is not just to label clothing as casual or formal. ClosetGraph should reason about how garments work together: layering, coverage, color, texture, dress-code range, weather, and context.

## Why It Is Interesting

A blazer does not automatically make every outfit professional. A tank can become business casual under the right layer. Tailored trousers can shift from office to dinner depending on the top and styling. ClosetGraph is designed around that kind of compositional outfit reasoning.

## Planned System

- LangGraph orchestration with a supervisor and specialist agents where justified.
- Persistent wardrobe memory separate from session conversation memory.
- Style-guide RAG over a custom styling corpus.
- Candidate outfit generation using only owned items.
- Deterministic validation for hard constraints like ownership.
- Model reasoning for nuanced outfit suitability and ranking.
- Groq as the primary free hosted model path, with Ollama as a local fallback.
- Later support for weather context, web research, photo-based wardrobe entry, tracing, evals, UI, and Hugging Face Spaces deployment.

## Current Status

ClosetGraph is being developed incrementally. Stages 1, 2, and 3 are complete.

**Stage 1 (walking skeleton):** a notebook demo (`run_demo.ipynb`) loads a small wardrobe, generates owned-item outfit candidates, and ranks them with a Groq-backed stylist, with a pytest suite over the core graph and stylist.

**Stage 2 (Style Guide RAG and Evals):**
- Style-guide RAG: a local Chroma index over a custom styling corpus, with stratified per-category retrieval wired into the stylist prompt.
- A deterministic validity checker for hard constraints (ownership, outfit structure).
- An evaluation harness (`closetgraph/eval_harness.py`) over a 24-case golden set, separating hard constraints, ranking preferences, and behavioral expectations; pairwise compositional suitability tests with context reversal; and an LLM-as-judge for subjective color-cohesion cases.
- Langfuse tracing around every model call. Findings, metrics, and a 20B-vs-120B model comparison are in [docs/eval-results.md](docs/eval-results.md).

**Stage 3 (Tool Context - weather):**
- Weather as a tool: optional temperature/condition, or a location resolved via Open-Meteo (keyless, stdlib only). User-provided weather takes precedence; a lookup fills only the fields the user left out; a failed lookup degrades gracefully without fabricating conditions.
- Deterministic gating, not an LLM router: weather is fetched when inputs are available; whether it changes the outfit is the stylist's judgment. Weather influences but never overrides the occasion's dress code.
- Conditional weather RAG: the weather corpus is retrieved only when weather is present. Weather evals (pairwise context-reversal plus golden cases) confirm the ranking shifts for defensible reasons - cold favors a coat, warm drops it, and a cold client meeting stays professional.
- Engineering lessons from Stages 2-3 are collected in [docs/evaluation_lessons.md](docs/evaluation_lessons.md).

### Observability

Every model call is traced to Langfuse on the free tier (zero cost):

![Langfuse dashboard showing traced runs at $0 cost](docs/langfuse-dashboard.png)

## Running It

- **Index (once, and after any corpus change):** `python -m closetgraph.indexing` builds the local Chroma index. The index is a generated artifact and is not committed, so build it before the first run; weather guidance is part of the corpus as of Stage 3.
- **Demo:** open `run_demo.ipynb` on the project virtualenv kernel to see ranked outfits, the retrieved style guidance, and (for occasions with weather) how weather shifts the ranking.
- **Weather:** pass a `WeatherInput` (explicit `temperature_c`/`condition`, or a `location` to look up) alongside the occasion. See "Run with real weather" below.
- **Evaluation:** the runner `scripts/run_evals.py` runs the golden, pairwise, and weather suites and writes a JSON report; the harness in `closetgraph/eval_harness.py` exposes them programmatically. These make live Groq calls, so throttle for the free-tier rate limit.

The initial handwritten design notes capture the early architecture, open questions, and decisions that shaped the roadmap:

[View initial design notes](docs/handwritten_notes_updated.pdf)

See [roadmap.md](roadmap.md) for the staged development plan.
