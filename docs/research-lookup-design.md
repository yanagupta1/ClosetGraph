# Research / Web-Lookup Path (design only)

Design note for a later stage. Nothing here is implemented; it records the
intended shape so the Stage 3 tool work composes cleanly with the Stage 4
research agent.

## Purpose

Some occasions reference things the style corpus does not cover: an unfamiliar
dress code ("Edwardian picnic"), a specific venue's norms, or an event whose
formality is ambiguous. A research lookup would fetch and summarize external
context into styling-relevant constraints, the same way weather is fetched and
summarized into a WeatherContext.

## Shape (mirrors the weather tool)

- A deterministic gate decides availability, not stylistic relevance: run
  research only when the occasion is flagged ambiguous or names an unknown
  venue/dress code. Whether the result changes the outfit stays the stylist's
  judgment, exactly as with weather.
- A fetcher (web search / page fetch) returns raw results; a summarizer node
  distills them into a small, structured `ResearchContext` (e.g. inferred
  formality range, notable constraints, a source URL for provenance).
- Graceful failure like weather: on lookup failure, surface a `research_error`
  and continue without fabricating constraints.
- The summarized context is injected into the stylist prompt as additional
  guidance, and may feed retrieval (a research-derived query).

## Boundaries

- Free-tier only; no paid search API. A keyless or free search path is a
  prerequisite before building this.
- Shopping is explicitly out of scope for the recommendation loop. Any future
  shopping use is limited to wardrobe-gap analysis, separate from ranking owned
  outfits.
- Natural-language extraction (turning "a gala in Chicago next Friday" into
  {occasion, location, date}) belongs with this research/understanding step, not
  the deterministic weather tool.

## Reuse from Stage 3

The patterns are already in place: deterministic availability gating, injected
fetchers for testability, graceful failure via an error field, and provenance on
resolved context. The research path should follow the same contracts.
