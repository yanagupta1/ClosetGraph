"""
closetgraph/retrieval.py
"""

import chromadb

# Always-retrieved categories. Weather is retrieved conditionally (only when a
# request carries weather context), so it is not in this list.
CATEGORIES = ["occasion", "layering", "garments", "color"]


def _weather_query(weather_context) -> str:
    """Build a weather-flavoured query so the weather chunks that match are the
    ones about the actual conditions, not whatever the occasion string implies."""
    parts = []
    if weather_context.condition:
        parts.append(weather_context.condition)
    if weather_context.temperature_c is not None:
        parts.append(f"{weather_context.temperature_c:g} degrees celsius")
    return " ".join(parts) or "weather"


def retrieve_style_guidance(
    occasion: str,
    persist_path: str = "./chroma_db",
    k_per_category: int = 2,
    weather_context=None,
) -> list[str]:
    """
    Stratified retrieval, NOT a single top-k over the whole collection.

    For each always-on category (occasion, layering, garments, color) query the
    occasion string. When weather_context is provided, additionally query the
    weather category with a weather-derived query - so weather guidance is
    retrieved ONLY when the request actually has weather, keeping indoor and
    no-weather runs identical to Stage 2. Category order is preserved (not
    re-sorted by score) for deterministic, debuggable output.
    """
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection("style_guide")

    guidance: list[str] = []
    for category in CATEGORIES:
        result = collection.query(
            query_texts=[occasion],
            n_results=k_per_category,
            where={"category": category},
        )
        guidance.extend((result.get("documents") or [[]])[0])

    if weather_context is not None:
        result = collection.query(
            query_texts=[_weather_query(weather_context)],
            n_results=k_per_category,
            where={"category": "weather"},
        )
        guidance.extend((result.get("documents") or [[]])[0])

    # Cap scales with how many categories were queried (4, or 5 with weather).
    categories_queried = len(CATEGORIES) + (1 if weather_context is not None else 0)
    return guidance[: k_per_category * categories_queried]
