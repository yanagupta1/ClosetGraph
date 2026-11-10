from pydantic import BaseModel
from typing import Literal, TypedDict


class WardrobeItem(BaseModel):
    id: str
    name: str  # free text, e.g. "black scoop-neck tank top"
    category: Literal["top", "bottom", "layering", "accessory"]


class OutfitCandidate(BaseModel):
    items: list[str]  # WardrobeItem ids


class RankedOutfit(BaseModel):
    items: list[str]
    rank: int
    reasoning: str


class WeatherInput(BaseModel):
    """Weather the user explicitly supplied with a request. All optional, so a
    request can carry no weather at all (Stage 2 behaviour), an explicit
    temperature/condition, or just a location to look up."""

    temperature_c: float | None = None
    condition: str | None = None  # free text, e.g. "rainy", "windy", "hot and humid"
    location: str | None = None  # used only to look weather up when not given directly


class WeatherContext(BaseModel):
    """Weather actually resolved and used downstream. Provenance is per-field so
    a context can be part user-provided and part fetched: `sources` maps each
    resolved field name ("temperature_c", "condition") to "user" or
    "open-meteo"."""

    temperature_c: float | None = None
    condition: str | None = None
    sources: dict[str, str]


class ResearchContext(BaseModel):
    """Background on an unfamiliar occasion/venue/dress code, distilled by the
    research agent into styling-relevant constraints, with provenance."""

    query: str  # what was looked up
    summary: str  # styling-relevant constraints distilled from the source
    inferred_formality: str | None = None  # e.g. "formal / black-tie", if clear
    source_title: str | None = None
    source_url: str | None = None


class ClosetGraphState(TypedDict):
    wardrobe: list[WardrobeItem]
    occasion: str
    candidate_outfits: list[OutfitCandidate]
    ranked_outfits: list[RankedOutfit]
    retrieved_guidance: list[str]  # style-guide chunks
    # Stage 3 weather (all optional; omitted keys keep Stage 2 requests working):
    weather_input: WeatherInput | None  # explicit user request, if any
    weather_context: WeatherContext | None  # resolved weather actually used, or None
    weather_relevant: bool  # whether the occasion/request warrants weather
    weather_error: str | None  # set when a lookup failed, so the failure is observable
    # Stage 4 research (all optional):
    research_needed: bool  # supervisor's routing decision
    research_query: str | None  # what the supervisor chose to look up
    research_reason: str | None  # why research was or was not needed (explainability)
    research_context: ResearchContext | None  # resolved research actually used, or None
    research_error: str | None  # set when a lookup/summary failed, so it is observable

