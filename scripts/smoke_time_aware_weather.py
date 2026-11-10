"""Targeted live smoke test for time-aware weather.

Run manually (needs a GROQ_API_KEY; Open-Meteo is keyless). Not collected by
pytest. Two parts:

  1. Two real-weather requests through the full graph: one date-only hint
     (daily forecast) and one dinner hint (hourly evening window). These prove
     the forecast path resolves end to end against the live Open-Meteo API.

  2. Three controlled ranking contrasts: the occasion and candidate outfits are
     held fixed and only the (time-resolved) weather changes, so any reordering
     is attributable to the weather. Weather is injected here, not fetched, so
     the contrast is deterministic in its inputs; the stylist call is live.

Usage:  python scripts/smoke_time_aware_weather.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from dotenv import load_dotenv

from closetgraph.candidates import generate_candidates
from closetgraph.graph import build_graph
from closetgraph.models import WardrobeItem, WeatherContext, WeatherInput
from closetgraph.stylist import rank_outfits

load_dotenv()
WARDROBE = [WardrobeItem(**i) for i in json.loads(Path("docs/wardrobe.json").read_text())]
NAMES = {w.id: w.name for w in WARDROBE}
LOCATION = "Chicago"


def _outfit(o):
    return ", ".join(NAMES.get(i, i) for i in o.items)


def real_request(occasion, when):
    print(f"\n=== REAL: {occasion!r}  when={when!r}  @ {LOCATION} ===")
    result = build_graph().invoke(
        {
            "wardrobe": WARDROBE,
            "occasion": occasion,
            "candidate_outfits": [],
            "ranked_outfits": [],
            "weather_input": WeatherInput(location=LOCATION, when=when),
        }
    )
    wc = result.get("weather_context")
    if wc:
        print(f"  resolved weather: {wc.temperature_c}C, {wc.condition}  sources={wc.sources}")
        print(f"  forecast_note: {wc.forecast_note}")
    if result.get("weather_error"):
        print(f"  weather_error: {result['weather_error']}")
    top = result["ranked_outfits"][0]
    print(f"  top outfit: {_outfit(top)}")
    print(f"    reasoning: {top.reasoning}")
    return {
        "occasion": occasion,
        "when": when,
        "weather": wc.model_dump() if wc else None,
        "weather_error": result.get("weather_error"),
        "top_outfit": {"items": top.items, "reasoning": top.reasoning},
    }


def contrast(title, occasion, label_a, wc_a, label_b, wc_b):
    print(f"\n=== CONTRAST: {title} ===")
    candidates = generate_candidates(WARDROBE)
    out = {"title": title, "occasion": occasion, "sides": {}}
    for label, wc in ((label_a, wc_a), (label_b, wc_b)):
        ranked = rank_outfits(candidates, WARDROBE, occasion, None, weather_context=wc)
        top = ranked[0]
        print(f"  [{label}] {wc.temperature_c}C {wc.condition} ({wc.sources})")
        print(f"      top: {_outfit(top)}")
        print(f"      why: {top.reasoning}")
        out["sides"][label] = {"items": top.items, "reasoning": top.reasoning}
    return out


results = {"real": [], "contrasts": []}

# Part 1: real weather through the graph.
results["real"].append(real_request("an outdoor weekend walk", "next Saturday"))
results["real"].append(real_request("a dinner out with friends", "dinner next Saturday"))

# Part 2: controlled ranking contrasts (weather injected, stylist live).
results["contrasts"].append(
    contrast(
        "same dinner: warm afternoon vs cold evening",
        "a dinner out with friends",
        "warm afternoon",
        WeatherContext(temperature_c=24, condition="clear", sources={"temperature_c": "open-meteo", "condition": "open-meteo"}, forecast_note="forecast for the afternoon"),
        "cold evening",
        WeatherContext(temperature_c=5, condition="clear", sources={"temperature_c": "open-meteo", "condition": "open-meteo"}, forecast_note="forecast for the evening"),
    )
)
results["contrasts"].append(
    contrast(
        "same brunch: dry morning vs rainy afternoon",
        "a casual weekend brunch",
        "dry morning",
        WeatherContext(temperature_c=15, condition="clear", sources={"temperature_c": "open-meteo", "condition": "open-meteo"}, forecast_note="forecast for the morning"),
        "rainy afternoon",
        WeatherContext(temperature_c=14, condition="rainy", sources={"temperature_c": "open-meteo", "condition": "open-meteo"}, forecast_note="forecast for the afternoon"),
    )
)
results["contrasts"].append(
    contrast(
        "same event: explicit user temp vs conflicting API temp",
        "an outdoor garden party",
        "user says 27C (override)",
        WeatherContext(temperature_c=27, condition="clear", sources={"temperature_c": "user", "condition": "open-meteo"}, forecast_note="forecast for the afternoon"),
        "API says 6C",
        WeatherContext(temperature_c=6, condition="clear", sources={"temperature_c": "open-meteo", "condition": "open-meteo"}, forecast_note="forecast for the afternoon"),
    )
)

Path("smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nwrote smoke_results.json; traces are in Langfuse")
