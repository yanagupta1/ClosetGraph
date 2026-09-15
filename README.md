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

This repo is in the planning stage. The current focus is defining the roadmap, architecture boundaries, and evaluation strategy before implementation begins.

See [roadmap.md](roadmap.md) for the staged development plan.
