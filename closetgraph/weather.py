"""Weather resolution: turn an optional WeatherInput into a resolved
WeatherContext (or nothing), with clear provenance and graceful failure.

The Open-Meteo fetch (fetch_open_meteo) is keyless and uses only the stdlib. It
is injected into resolve_weather as a `fetcher` callable so the resolution logic
stays pure and testable without any external API call.

LIMITATION - time awareness: fetch_open_meteo returns CURRENT conditions at the
location. ClosetGraph does not support time-aware forecasts, so weather for a
future occasion (e.g. "dinner next Friday") is NOT resolved to that time. We do
not silently substitute current conditions for a future request; callers must
treat an open-meteo WeatherContext as "current conditions", not a forecast.
"""
import json
import logging
from urllib.parse import urlencode
from urllib.request import urlopen

from closetgraph.models import WeatherContext, WeatherInput

logger = logging.getLogger(__name__)

# A fetcher takes a location string and returns {"temperature_c": float|None,
# "condition": str|None}, or raises / returns None if it cannot resolve.
Fetcher = "Callable[[str], dict | None]"

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HTTP_TIMEOUT = 10

# WMO weather codes -> a coarse condition word the stylist/corpus can reason over.
_WMO_CONDITIONS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "cloudy",
    45: "foggy",
    48: "foggy",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    61: "rainy",
    63: "rainy",
    65: "rainy",
    66: "rainy",
    67: "rainy",
    71: "snowy",
    73: "snowy",
    75: "snowy",
    77: "snowy",
    80: "rainy",
    81: "rainy",
    82: "rainy",
    85: "snowy",
    86: "snowy",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def _get_json(url):
    with urlopen(url, timeout=_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_open_meteo(location: str) -> dict:
    """
    Resolve CURRENT conditions for a location via Open-Meteo (keyless, stdlib):
    geocode the location name to coordinates, then read current temperature and
    weather code. Returns {"temperature_c": float, "condition": str}.

    Raises on an unknown location, network error, or malformed response so the
    caller (resolve_weather) can surface the failure through weather_error. Never
    invents a location or fabricates conditions.

    Time awareness: this is CURRENT weather only, not a forecast for a future
    occasion time (see the module docstring).
    """
    geo = _get_json(f"{_GEOCODE_URL}?{urlencode({'name': location, 'count': 1})}")
    results = geo.get("results") or []
    if not results:
        raise ValueError(f"no geocoding match for location {location!r}")
    match = results[0]
    if "latitude" not in match or "longitude" not in match:
        raise ValueError(f"geocoding result missing coordinates for {location!r}")

    params = {
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "current": "temperature_2m,weather_code",
    }
    forecast = _get_json(f"{_FORECAST_URL}?{urlencode(params)}")
    current = forecast.get("current") or {}
    if "temperature_2m" not in current:
        raise ValueError("forecast response missing current temperature")

    return {
        "temperature_c": current["temperature_2m"],
        "condition": _WMO_CONDITIONS.get(current.get("weather_code"), "unknown"),
    }


def resolve_weather(weather_input: WeatherInput | None, fetcher=None):
    """
    Resolve weather for a request, returning (context, error):

      - context: a WeatherContext when any field was resolved, else None.
      - error:   a short string when a lookup was attempted and failed, else None.

    Per-field merge, not all-or-nothing:
      - No input, or nothing usable -> (None, None). Not an error; the request
        simply carries no weather (Stage 2 behaviour).
      - User-provided fields always win and are tagged "user". The fetcher is
        only consulted to fill fields the user left out (and only when a
        location is available), so e.g. "3C in NYC, no condition" keeps the
        user's temperature and fetches just the condition.
      - A lookup is only attempted when a field is missing AND a location is
        given. On failure we never fabricate: any user-provided fields are kept,
        the missing ones stay None, and the failure is surfaced via error.
    """
    if weather_input is None:
        return None, None

    temperature_c = weather_input.temperature_c
    condition = weather_input.condition
    sources: dict[str, str] = {}
    if temperature_c is not None:
        sources["temperature_c"] = "user"
    if condition is not None:
        sources["condition"] = "user"

    error = None
    missing = temperature_c is None or condition is None
    if missing and weather_input.location:
        if fetcher is None:
            error = f"no weather fetcher available to resolve {weather_input.location!r}"
            logger.warning(error)
        else:
            try:
                fetched = fetcher(weather_input.location)
            except Exception as exc:  # never fabricate weather on failure
                error = f"weather lookup failed for {weather_input.location!r}: {exc!r}"
                logger.warning(error)
                fetched = None
            if fetched is None and error is None:
                error = f"weather lookup returned nothing for {weather_input.location!r}"
                logger.warning(error)
            if fetched:
                # Fill only gaps; user-provided fields are never overwritten.
                if temperature_c is None and fetched.get("temperature_c") is not None:
                    temperature_c = fetched["temperature_c"]
                    sources["temperature_c"] = "open-meteo"
                if condition is None and fetched.get("condition") is not None:
                    condition = fetched["condition"]
                    sources["condition"] = "open-meteo"

    if temperature_c is None and condition is None:
        return None, error

    return (
        WeatherContext(
            temperature_c=temperature_c, condition=condition, sources=sources
        ),
        error,
    )
